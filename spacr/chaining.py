"""Auto-chaining: a module's inputs default to where the last run *actually* wrote.

Opening Measure has always meant typing the plate folder again, and getting it
wrong meant a twenty-minute run against the previous plate.  The path was
never a mystery — Mask had just written it — but nothing carried the answer
across, so every module screen started from ``"path"``.

:mod:`spacr.ports` now declares what each module consumes and produces, and
:mod:`spacr.artifacts` records where every finished run put its outputs.  This
module joins the two:

* :func:`chained_inputs` asks the registry — ``latest(kind, project=…)`` —
  where the upstream module's output *is*, and turns that into the value a
  settings key should hold.  The answer comes from the row the producer wrote,
  never from re-deriving ``<root>/merged`` and hoping;
* :func:`resolve_settings` applies those values to a settings dict **without
  ever overwriting a path the user edited**.  A user edit is remembered in a
  :class:`PinStore` and wins forever after; when the upstream later moves, the
  new location is *offered* (:class:`HeldPin`) rather than pushed;
* :func:`staleness_notes` turns :meth:`spacr.artifacts.Registry.is_stale` into
  sentences a user can act on, keyed by the cause codes so the reason is
  specific — "Mask ran again after this" is a different problem from "the
  settings changed";
* :func:`next_steps` answers "this run finished — now what?", using
  :func:`spacr.ports.next_modules` for the candidates,
  :func:`chained_inputs` for their pre-filled settings and
  :func:`spacr.ports.check_ready` so a successor that *cannot* run is offered
  with its blocking reason rather than silently.

Nothing here imports Qt, numpy or torch: the Qt layer
(:mod:`spacr.qt.chaining`) is a thin skin over these functions, and the same
answers are available to the CLI and to a batch runner.

Public API
----------
``PinStore``, ``pin_store``, ``state_path``
    The memory of which paths the user edited by hand.
``Binding``, ``BINDINGS``, ``register_binding``, ``binding_for``
    Which settings key an input port fills, and in what form.
``ChainedInput``, ``chained_inputs``, ``resolve_settings``, ``Resolution``
    Auto-chaining itself.
``StaleNote``, ``stale_inputs``, ``stale_outputs``, ``staleness_notes``
    Staleness, said out loud.
``NextStep``, ``next_steps``
    "Continue to the next step."
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from . import artifacts as _artifacts
from . import ports as _ports
from .artifacts import (CAUSE_CYCLE, CAUSE_SETTINGS_CHANGED, CAUSE_UNKNOWN,
                        CAUSE_UPSTREAM_MISSING, CAUSE_UPSTREAM_NEWER,
                        CAUSE_UPSTREAM_STALE, CAUSE_UPSTREAM_SUPERSEDED,
                        Artifact, Registry, Staleness)
from .ports import Port, Readiness, ResolvedPort
from .validate import ALT_SRC_KEYS, APP_ALIASES

__all__ = [
    "BINDINGS",
    "Binding",
    "CAUSE_FIX",
    "CAUSE_TEXT",
    "ChainedInput",
    "DB_SUFFIXES",
    "DropChoice",
    "DropResolution",
    "DropTarget",
    "FROM_LAYOUT",
    "FROM_REGISTRY",
    "HeldPin",
    "MAX_CHILDREN",
    "MAX_CLIMB",
    "NextStep",
    "PATH",
    "PIN_STATE_ENV",
    "PLACEHOLDER_PATHS",
    "PROJECT",
    "PinStore",
    "ROOT",
    "Resolution",
    "StaleNote",
    "TABLE_SUFFIXES",
    "binding_for",
    "candidate_roots",
    "chained_inputs",
    "db_candidates",
    "explain_causes",
    "is_empty_path",
    "layout_directories",
    "looks_laid_out",
    "next_steps",
    "pin_store",
    "placeholder_paths",
    "ports_for_kinds",
    "project_root_of",
    "register_binding",
    "resolve_drop",
    "resolve_settings",
    "result_tables",
    "same_path",
    "satisfies",
    "source_key",
    "stale_inputs",
    "stale_outputs",
    "staleness_notes",
    "state_path",
]


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

#: The strings the shipped settings dicts use to mean "no folder chosen yet".
#: ``set_default_settings_preprocess_generate_masks`` writes ``"path"``; the
#: Qt empty-state banner and the live-preview autoload already treat these
#: four as empty, and auto-chaining has to agree with them or it would refuse
#: to fill a field that looks filled.
PLACEHOLDER_PATHS: Tuple[str, ...] = (
    "path", "/path", "/path/to/src", "list of paths", "path to images",
)


def placeholder_paths() -> Tuple[str, ...]:
    """Return the values that mean "nothing chosen yet"."""
    return PLACEHOLDER_PATHS


def is_empty_path(value: Any) -> bool:
    """True when a settings value holds no real path.

    :param value: the current value of a path-ish settings key.
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not [item for item in value if not is_empty_path(item)]
    text = str(value).strip()
    return not text or text in PLACEHOLDER_PATHS


def same_path(left: Any, right: Any) -> bool:
    """Compare two settings values as paths, list-insensitively.

    ``["/plate"]`` and ``"/plate"`` name the same folder; Classify keeps its
    source in a list and every other module keeps it as a string, so a
    comparison that called those different would record a pin every time a
    Classify screen was seeded with its own auto-chained value.
    """
    def flatten(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out: List[str] = []
            for item in value:
                out.extend(flatten(item))
            return out
        text = str(value).strip()
        return [os.path.normpath(text)] if text else []

    return flatten(left) == flatten(right)


# ---------------------------------------------------------------------------
# The pin store — which paths the user typed by hand
# ---------------------------------------------------------------------------

#: Points the pin file somewhere else. Set by tests, and by a portable or
#: multi-user install that keeps per-user state off the home directory.
PIN_STATE_ENV = "SPACR_CHAINING_PINS"


def state_path() -> str:
    """Return the file the user's pinned paths live in.

    ``$SPACR_CHAINING_PINS`` wins; otherwise XDG state storage, matching
    :func:`spacr.remote_execution.state_directory` rather than inventing a
    second convention for the same kind of data.

    :returns: an absolute path. The file need not exist.
    """
    override = os.environ.get(PIN_STATE_ENV, "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = (os.path.join(os.path.expanduser(xdg), "spacr")
            if xdg else
            os.path.join(os.path.expanduser("~"), ".local", "state", "spacr"))
    return os.path.join(root, "chaining", "pins.json")


class PinStore:
    """The paths a user edited by hand, remembered across restarts.

    Auto-chaining is only welcome while it is filling in a blank.  The moment
    a user types a path of their own, that path is theirs: it survives a
    reopen, a restart, and every subsequent upstream run.  This is the record
    that makes that true, and it is deliberately *not* the settings dict —
    a settings dict cannot distinguish "the user chose this" from "we put it
    there".

    Read lazily and written through a temporary file plus :func:`os.replace`,
    so a crash mid-write cannot leave a half-written JSON file that would lose
    every pin at once.

    :param path: the JSON file. Defaults to :func:`state_path`.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(os.path.expanduser(path or state_path()))
        self._data: Optional[Dict[str, Dict[str, Any]]] = None

    # -- storage ----------------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Return the in-memory pin table, reading the file on first use."""
        if self._data is not None:
            return self._data
        data: Dict[str, Dict[str, Any]] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            # No file, an unreadable one, or one someone hand-edited into
            # invalid JSON. A lost pin costs one re-typed path; refusing to
            # open the screen would cost the whole session.
            raw = {}
        if isinstance(raw, Mapping):
            for module, entries in raw.items():
                if isinstance(entries, Mapping):
                    data[str(module)] = {str(k): v for k, v in entries.items()}
        self._data = data
        return data

    def _save(self) -> None:
        """Write the pin table atomically, and never raise."""
        data = self._load()
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False,
                dir=os.path.dirname(self.path) or ".", suffix=".tmp")
            try:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, self.path)
        except OSError:
            # A read-only home, a full disk, a locked file on Windows. The
            # pin still holds for this session; it simply will not outlive it.
            pass

    def reload(self) -> "PinStore":
        """Drop the cached table so the next read hits the file.

        :returns: self, for chaining.
        """
        self._data = None
        return self

    # -- the pins ---------------------------------------------------------

    def pin(self, module: str, setting: str, value: Any) -> None:
        """Record that the user chose ``value`` for ``module``'s ``setting``.

        An empty value **removes** the pin rather than storing a blank one:
        clearing the field is how a user asks for the automatic default back,
        and a stored empty string would instead mean "the user chose nothing"
        and suppress chaining forever.

        :param module: module key.
        :param setting: settings key, e.g. ``"src"``.
        :param value: the path (or list of paths) the user entered.
        """
        key = _canonical(module)
        if is_empty_path(value):
            self.unpin(key, setting)
            return
        data = self._load()
        data.setdefault(key, {})[str(setting)] = value
        self._save()

    def unpin(self, module: str, setting: str) -> bool:
        """Forget the pin for one setting.

        :param module: module key.
        :param setting: settings key.
        :returns: True when there was one to forget.
        """
        data = self._load()
        entries = data.get(_canonical(module))
        if not entries or str(setting) not in entries:
            return False
        entries.pop(str(setting))
        if not entries:
            data.pop(_canonical(module), None)
        self._save()
        return True

    def pinned(self, module: str, setting: str) -> Any:
        """Return the pinned value, or None when the user never set one.

        :param module: module key.
        :param setting: settings key.
        """
        return self._load().get(_canonical(module), {}).get(str(setting))

    def pins(self, module: str) -> Dict[str, Any]:
        """Return every pin for one module, as a copy.

        :param module: module key.
        """
        return dict(self._load().get(_canonical(module), {}))

    def clear(self, module: str = "") -> None:
        """Forget one module's pins, or all of them.

        :param module: module key, or ``""`` for every module.
        """
        data = self._load()
        if module:
            data.pop(_canonical(module), None)
        else:
            data.clear()
        self._save()


_STORE: Optional[PinStore] = None


def pin_store(path: Optional[str] = None, *, refresh: bool = False) -> PinStore:
    """Return the process-wide :class:`PinStore`.

    :param path: use a specific file instead of :func:`state_path`. Passing
        one always builds a fresh store rather than handing back a cached one
        pointed at a different file.
    :param refresh: rebuild the shared store, re-reading :func:`state_path`.
        Needed after ``$SPACR_CHAINING_PINS`` changes, which is exactly what a
        test that isolates the state does.
    """
    global _STORE
    if path is not None:
        return PinStore(path)
    if _STORE is None or refresh or _STORE.path != os.path.abspath(
            os.path.expanduser(state_path())):
        _STORE = PinStore()
    return _STORE


# ---------------------------------------------------------------------------
# Bindings — which settings key an input port fills
# ---------------------------------------------------------------------------

#: The setting names the *project root* the artifact belongs to. This is what
#: ``src`` is for every pipeline module: the ports resolve relative to it.
ROOT = "root"
#: The setting names the artifact's own path — the file or folder itself.
PATH = "path"


@dataclass(frozen=True)
class Binding:
    """How one consumed port becomes one settings value.

    :param module: the consuming module key.
    :param role: the port's role within that module, e.g. ``"merged"``.
    :param setting: the settings key it fills, e.g. ``"src"``.
    :param form: :data:`ROOT` (the project the artifact belongs to) or
        :data:`PATH` (the artifact itself).
    """

    module: str
    role: str
    setting: str
    form: str = ROOT


#: Explicit bindings, keyed by ``(module, role)``. Only for the ports whose
#: settings key is *not* the module's source folder; everything else is
#: derived, so a module that joins :data:`spacr.ports.PORTS` chains without
#: an entry here.
BINDINGS: Dict[Tuple[str, str], Binding] = {}


def register_binding(binding: Binding, *, overwrite: bool = False) -> Binding:
    """Declare that one port fills one settings key.

    The seam a module with an unusual input key uses, so this table never has
    to be edited by hand.

    :param binding: the declaration.
    :param overwrite: allow replacing an existing one. Off by default, so two
        contributors claiming one port is an error rather than last-one-wins.
    :returns: the stored binding.
    :raises ValueError: on an empty field, an unknown form, or a duplicate.
    """
    module = _canonical(binding.module)
    role = str(binding.role).strip()
    setting = str(binding.setting).strip()
    if not module or not role or not setting:
        raise ValueError("a binding needs a module, a role and a setting")
    if binding.form not in (ROOT, PATH):
        raise ValueError(
            f"binding {module}.{role}: unknown form {binding.form!r}; "
            f"use {ROOT!r} or {PATH!r}")
    key = (module, role)
    if key in BINDINGS and not overwrite:
        raise ValueError(
            f"{module}.{role} is already bound to "
            f"{BINDINGS[key].setting!r}; pass overwrite=True to replace it")
    stored = Binding(module, role, setting, binding.form)
    BINDINGS[key] = stored
    return stored


def _canonical(module: str) -> str:
    """Return the canonical module key for ``module`` or an alias of it."""
    key = str(module).strip().lower()
    return APP_ALIASES.get(key, key)


def source_key(module: str) -> str:
    """Return the settings key naming ``module``'s source folder.

    The same lookup :mod:`spacr.ports` and :mod:`spacr.validate` make:
    :data:`spacr.ports.ROOT_KEYS` first (a module whose *output* names the
    project), then :data:`spacr.validate.ALT_SRC_KEYS`, then ``src``.

    :param module: module key or alias.
    """
    key = _canonical(module)
    return _ports.ROOT_KEYS.get(key) or ALT_SRC_KEYS.get(key, "src")


def binding_for(module: str, port: Port) -> Binding:
    """Return the binding for one consumed port, declared or derived.

    Derived means: the port fills the module's source folder, in
    :data:`ROOT` form.  That is the shipped truth for every pipeline module —
    Measure, Classify, UMAP and the rest all take a plate folder and find
    ``merged/`` or ``measurements/`` inside it — so the common case needs no
    declaration and a new module joins the chain the moment it declares ports.

    :param module: the consuming module key or alias.
    :param port: one of that module's consumed ports.
    """
    key = _canonical(module)
    declared = BINDINGS.get((key, port.role))
    if declared is not None:
        return declared
    return Binding(key, port.role, source_key(key), ROOT)


# ---------------------------------------------------------------------------
# Resolving an artifact into a settings value
# ---------------------------------------------------------------------------

def _artifact_root(artifact: Artifact, port: Port) -> str:
    """Return the project root ``artifact`` belongs to.

    The project the *producer recorded*, which is the whole point: it is where
    the upstream run says it worked, not where this module's path convention
    would have guessed.  The convention is used only when a registration
    carried no project at all — a registry opened without one — and then it is
    stripped from the artifact's own path rather than rebuilt from a settings
    key, so the answer still comes from the artifact.

    :param artifact: the registered upstream output.
    :param port: the port it satisfies, whose declared relative path says how
        far above the artifact the project root sits.
    """
    if artifact.project:
        return artifact.project
    root = artifact.path
    for _ in [p for p in os.path.normpath(port.path).split(os.sep) if p and p != "."]:
        root = os.path.dirname(root)
    return root


def _value_for(artifact: Artifact, port: Port, binding: Binding,
               current: Any) -> Any:
    """Return the settings value ``artifact`` should produce.

    :param artifact: the registered upstream output.
    :param port: the consumed port it satisfies.
    :param binding: how the port maps onto a settings key.
    :param current: the key's current value, which decides whether the answer
        is wrapped in a list — Classify keeps its sources in one and every
        other module does not.
    """
    value = (artifact.path if binding.form == PATH
             else _artifact_root(artifact, port))
    if isinstance(current, (list, tuple)):
        return [value]
    return value


# ---------------------------------------------------------------------------
# Finding the registry to ask
# ---------------------------------------------------------------------------

def candidate_roots(module: str,
                    settings: Optional[Mapping[str, Any]] = None,
                    *, root: str = "",
                    roots: Sequence[str] = ()) -> Tuple[str, ...]:
    """Return the project roots to search for ``module``'s inputs, in order.

    The module's own source folder first — a user who has already named a
    plate means that plate — then whatever the caller offers.  The Qt layer
    supplies the folders the upstream modules last ran in, which is how
    opening Measure on a blank screen finds the plate Mask just finished.

    :param module: module key or alias.
    :param settings: the settings dict being edited.
    :param root: an explicit root, tried first.
    :param roots: further candidates, in preference order.
    :returns: absolute, de-duplicated, existing-or-not (existence is the
        registry's problem, not this function's).
    """
    ordered: List[str] = []
    for candidate in (root, _ports.project_root(settings, module), *roots):
        if not candidate:
            continue
        absolute = _ports.project_root(candidate, module)
        if absolute and absolute not in ordered:
            ordered.append(absolute)
    return tuple(ordered)


def _registry_for(root: str, registry: Optional[Registry]) -> Optional[Registry]:
    """Return the registry covering ``root``, or None when there is none.

    Never *creates* one: asking "what did the last run write here?" must not
    leave an empty ``artifacts.db`` in a folder the user was only browsing.
    """
    if registry is not None:
        return registry
    try:
        path = _artifacts.registry_path(root)
    except ValueError:
        return None
    if not os.path.isfile(path):
        return None
    try:
        return _artifacts.open_registry(root, create=False)
    except (FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Auto-chaining
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainedInput:
    """One input a module can take from a run that already happened.

    :param module: the consuming module key.
    :param setting: the settings key this fills.
    :param role: the consumed port's role.
    :param kind: the :mod:`spacr.ports` kind, e.g. ``"merged-arrays"``.
    :param value: what the settings key should become.
    :param artifact: the registered upstream output it came from.
    :param producer: the module that wrote it.
    :param root: the project root the artifact belongs to.
    :param staleness: the artifact's own staleness, so a chained default can
        say "yes, and it is out of date" instead of quietly handing over a
        stale folder.
    :param required: whether the consuming module needs this port.
    """

    module: str
    setting: str
    role: str
    kind: str
    value: Any
    artifact: Artifact
    producer: str
    root: str
    staleness: Optional[Staleness] = None
    required: bool = True

    @property
    def path(self) -> str:
        """The artifact's own path — where the producer really wrote."""
        return self.artifact.path

    @property
    def stale(self) -> bool:
        """True when the registry reports this input as out of date."""
        return bool(self.staleness and self.staleness.stale)

    def describe(self) -> str:
        """One line: what this is, and who made it."""
        return (f"{self.kind} from {self.producer} at {self.artifact.path}")


def chained_inputs(module: str,
                   settings: Optional[Mapping[str, Any]] = None,
                   *,
                   root: str = "",
                   roots: Sequence[str] = (),
                   registry: Optional[Registry] = None,
                   check_staleness: bool = True) -> Tuple[ChainedInput, ...]:
    """Return where ``module``'s inputs actually are, one entry per port.

    For every port ``module`` consumes, the registry is asked for the newest
    artifact of that kind in each candidate project, in order, and the first
    hit wins.  The answer is the row the *producer* wrote — its path, its
    project, its settings hash — so a plate whose merged arrays ended up
    somewhere unusual chains correctly, which re-deriving ``<root>/merged``
    never could.

    Ports whose settings key the module does not have are skipped: inventing
    a key would put a value somewhere nothing reads.

    :param module: module key or alias.
    :param settings: the settings dict being edited. Used for the current
        values (list-or-string, and "is it already filled?") and for the
        module's own project root.
    :param root: an explicit project root, searched first.
    :param roots: further project roots to search, in preference order.
    :param registry: an open registry to ask instead of each project's own.
    :param check_staleness: also report whether each input is out of date.
        Costs one recursive query per input; off for a caller that only wants
        the paths.
    :returns: one :class:`ChainedInput` per resolved port, in declaration
        order.
    :raises spacr.ports.UnknownModule: when ``module`` declares no ports.
    """
    spec = _ports.module_ports(module)
    search = candidate_roots(spec.key, settings, root=root, roots=roots)
    # One registry per root for the whole call: opening one runs the schema
    # DDL, and a module with three ports across three candidate roots would
    # otherwise pay for nine of them on every keystroke.
    stores: Dict[str, Optional[Registry]] = {}
    found: List[ChainedInput] = []
    for port in spec.consumes:
        binding = binding_for(spec.key, port)
        if settings is not None and binding.setting not in settings:
            continue
        current = None if settings is None else settings.get(binding.setting)
        for candidate in search:
            if candidate not in stores:
                stores[candidate] = _registry_for(candidate, registry)
            store = stores[candidate]
            if store is None:
                continue
            artifact = store.latest(port.kind, project=candidate)
            if artifact is None:
                continue
            staleness = (store.is_stale(artifact.artifact_id)
                         if check_staleness else None)
            found.append(ChainedInput(
                module=spec.key, setting=binding.setting, role=port.role,
                kind=port.kind,
                value=_value_for(artifact, port, binding, current),
                artifact=artifact, producer=artifact.module,
                root=_artifact_root(artifact, port), staleness=staleness,
                required=port.required))
            break
    return tuple(found)


@dataclass(frozen=True)
class HeldPin:
    """A settings key auto-chaining did **not** touch, because the user owns it.

    :param setting: the settings key.
    :param value: the value the user chose, which is what the key holds.
    :param offered: the value auto-chaining would have used, or None when the
        registry has nothing to offer.
    :param chained: the :class:`ChainedInput` behind ``offered``.
    """

    setting: str
    value: Any
    offered: Any = None
    chained: Optional[ChainedInput] = None

    @property
    def differs(self) -> bool:
        """True when the upstream has moved away from the user's choice.

        The one case worth a word in the interface: the pin still wins, and
        the new location is offered beside it.
        """
        return (self.chained is not None
                and not same_path(self.value, self.offered))

    def describe(self) -> str:
        """One line for an interface that wants to offer the alternative."""
        if not self.differs or self.chained is None:
            return f"{self.setting} is set to {self.value!r}"
        return (f"{self.chained.producer} now writes "
                f"{self.chained.kind} to {self.chained.artifact.path}; "
                f"{self.setting} is pinned to {self.value!r}")


@dataclass(frozen=True)
class Resolution:
    """What auto-chaining did to a settings dict.

    :param module: the module key.
    :param settings: a **new** dict — the input is never mutated, because a
        caller that shows a diff needs both sides.
    :param filled: settings key → the chained input that filled it.
    :param held: settings key → the pin that stopped it being filled.
    :param inputs: every chained input found, filling or not.
    """

    module: str
    settings: Dict[str, Any]
    filled: Dict[str, ChainedInput] = field(default_factory=dict)
    held: Dict[str, HeldPin] = field(default_factory=dict)
    inputs: Tuple[ChainedInput, ...] = ()

    @property
    def moved(self) -> Tuple[HeldPin, ...]:
        """Pins whose upstream has since moved somewhere else."""
        return tuple(pin for pin in self.held.values() if pin.differs)


def resolve_settings(module: str,
                     settings: Mapping[str, Any],
                     *,
                     root: str = "",
                     roots: Sequence[str] = (),
                     registry: Optional[Registry] = None,
                     pins: Optional[PinStore] = None,
                     check_staleness: bool = True) -> Resolution:
    """Fill ``module``'s input paths from the registry, respecting user edits.

    The precedence, which is the whole design:

    1. **a pinned value wins.**  If the user has ever typed a path for this
       key, it is restored and nothing overwrites it — not a newer upstream
       run, not a different plate, not a restart.  When the upstream has moved
       since, the move is reported in :attr:`Resolution.held` for an interface
       to *offer*;
    2. otherwise, a value already in ``settings`` that is not a placeholder
       wins.  Loading a settings CSV, dropping a folder or seeding from
       another screen all land here, and none of them should be second-guessed
       within the same session;
    3. otherwise the registry's answer is used.

    :param module: module key or alias.
    :param settings: the settings dict to resolve. Not mutated.
    :param root: explicit project root, searched first.
    :param roots: further project roots, in preference order.
    :param registry: an open registry to ask instead of each project's own.
    :param pins: the user's pinned paths. Defaults to :func:`pin_store`.
    :param check_staleness: also report whether each input is out of date.
    :returns: a :class:`Resolution`.
    :raises spacr.ports.UnknownModule: when ``module`` declares no ports.
    """
    spec = _ports.module_ports(module)
    store = pins if pins is not None else pin_store()
    resolved = dict(settings)

    # A pin is restored BEFORE the lookup, so the candidate roots include the
    # plate the user pinned. Chaining Measure's crops off a pinned src is the
    # whole point of pinning it.
    #
    # Only the keys an input port binds to are consulted: a pin exists because
    # auto-chaining offered to fill that key, and restoring one for a key
    # nothing chains would let a stale state file quietly override a setting
    # the user is looking at.
    bound = {binding_for(spec.key, port).setting for port in spec.consumes}
    pinned_values: Dict[str, Any] = {}
    for key in [k for k in resolved if k in bound]:
        value = store.pinned(spec.key, key)
        if value is not None and not is_empty_path(value):
            pinned_values[key] = value
            resolved[key] = ([value] if isinstance(resolved.get(key),
                                                   (list, tuple))
                             and not isinstance(value, (list, tuple))
                             else value)

    inputs = chained_inputs(spec.key, resolved, root=root, roots=roots,
                            registry=registry,
                            check_staleness=check_staleness)

    filled: Dict[str, ChainedInput] = {}
    held: Dict[str, HeldPin] = {}
    seen: set = set()
    for chained in inputs:
        if chained.setting in seen:
            continue
        seen.add(chained.setting)
        if chained.setting in pinned_values:
            held[chained.setting] = HeldPin(
                setting=chained.setting, value=resolved[chained.setting],
                offered=chained.value, chained=chained)
            continue
        if not is_empty_path(resolved.get(chained.setting)):
            continue
        resolved[chained.setting] = chained.value
        filled[chained.setting] = chained

    # A pin with nothing to chain against is still held: the interface should
    # say the value is the user's, not that it came from a run.
    for key, value in pinned_values.items():
        held.setdefault(key, HeldPin(setting=key, value=resolved[key]))

    return Resolution(module=spec.key, settings=resolved, filled=filled,
                      held=held, inputs=inputs)


# ---------------------------------------------------------------------------
# Staleness, said out loud
# ---------------------------------------------------------------------------

#: One sentence per :mod:`spacr.artifacts` cause code, in the user's terms.
#: The cause is what makes the warning actionable — "re-run Mask" and "you
#: changed a setting" call for different actions, and a single "this is out of
#: date" would leave the user to guess which.
CAUSE_TEXT: Dict[str, str] = {
    CAUSE_UPSTREAM_MISSING:
        "an input it was made from is no longer in the registry",
    CAUSE_UPSTREAM_NEWER:
        "an input was produced again after this was made",
    CAUSE_UPSTREAM_SUPERSEDED:
        "a newer run has replaced one of its inputs",
    CAUSE_UPSTREAM_STALE:
        "one of its inputs is itself out of date",
    CAUSE_SETTINGS_CHANGED:
        "the settings on this screen differ from the ones that produced it",
    CAUSE_UNKNOWN:
        "it is not in the registry, so nothing is known about it",
    CAUSE_CYCLE:
        "its provenance refers back to itself and was not followed further",
}

#: What to do about each cause. Paired with :data:`CAUSE_TEXT` so a warning
#: never states a problem without an action, which is the contract
#: :class:`spacr.validate.Problem` already holds every settings warning to.
CAUSE_FIX: Dict[str, str] = {
    CAUSE_UPSTREAM_MISSING:
        "Re-run {producer} on this project, then re-run {module}.",
    CAUSE_UPSTREAM_NEWER: "Re-run {module} so it uses the new input.",
    CAUSE_UPSTREAM_SUPERSEDED: "Re-run {module} against the newer input.",
    CAUSE_UPSTREAM_STALE:
        "Re-run the steps above {module} first, then {module}.",
    CAUSE_SETTINGS_CHANGED:
        "Re-run {module} with these settings, or restore the settings that "
        "produced the existing result.",
    CAUSE_UNKNOWN: "Re-run {module} so the result is recorded.",
    CAUSE_CYCLE: "Report this: a provenance cycle should not be possible.",
}


def explain_causes(causes: Iterable[str]) -> str:
    """Render staleness cause codes as one readable clause.

    :param causes: cause codes from :attr:`spacr.artifacts.Staleness.causes`.
    :returns: the sentences joined with "; ", de-duplicated in first-seen
        order. Unknown codes are passed through as themselves rather than
        dropped — a code this table has not caught up with is still a fact.
    """
    seen: List[str] = []
    for cause in causes:
        text = CAUSE_TEXT.get(cause, str(cause))
        if text not in seen:
            seen.append(text)
    return "; ".join(seen)


@dataclass(frozen=True)
class StaleNote:
    """One out-of-date artifact, with the reason and the fix.

    :param module: the module whose screen this is being shown on.
    :param direction: ``"input"`` (something this module reads is stale) or
        ``"output"`` (a result this module already produced is stale).
    :param kind: the :mod:`spacr.ports` kind.
    :param role: the port role.
    :param path: where the artifact is.
    :param producer: the module that wrote it.
    :param artifact_id: the registry id.
    :param causes: the machine cause codes, verbatim from
        :class:`spacr.artifacts.Staleness`.
    :param reasons: the registry's own sentences, kept because they name the
        specific upstream path that moved.
    :param missing: the artifact's file is gone, which is an availability
        problem rather than a provenance one.
    """

    module: str
    direction: str
    kind: str
    role: str
    path: str
    producer: str
    artifact_id: str
    causes: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    missing: bool = False

    @property
    def headline(self) -> str:
        """One line naming what is stale and why."""
        what = ("The {kind} this run produced" if self.direction == "output"
                else "The {kind} this run would read").format(kind=self.kind)
        return f"{what} is out of date: {explain_causes(self.causes)}."

    @property
    def fix(self) -> str:
        """What to do about it, in the user's terms."""
        template = CAUSE_FIX.get(
            self.causes[0] if self.causes else CAUSE_UNKNOWN,
            "Re-run {module}.")
        return template.format(module=self.module,
                               producer=self.producer or "the previous step")

    @property
    def detail(self) -> str:
        """The registry's own sentences, which name the paths involved."""
        return "; ".join(self.reasons)

    def to_problem(self):
        """Return this note as a :class:`spacr.validate.Problem`.

        So a caller can print staleness through
        :func:`spacr.validate.format_report` beside the settings pre-flight
        and the port readiness check, rather than inventing a third format.
        """
        from .validate import WARNING, Problem
        return Problem(WARNING, self.role, self.headline, self.fix)


def _note(module: str, direction: str, resolved: ResolvedPort,
          artifact: Artifact, staleness: Staleness) -> StaleNote:
    """Build one :class:`StaleNote` from a registry answer."""
    return StaleNote(
        module=module, direction=direction, kind=artifact.kind,
        role=resolved.role, path=artifact.path, producer=artifact.module,
        artifact_id=artifact.artifact_id, causes=tuple(staleness.causes),
        reasons=tuple(staleness.reasons), missing=staleness.missing)


def _walk(module: str, direction: str,
          resolved_ports: Sequence[ResolvedPort],
          settings: Optional[Mapping[str, Any]],
          store: Registry, compare_settings: bool) -> Tuple[StaleNote, ...]:
    """Ask the registry about each port and keep the stale answers."""
    notes: List[StaleNote] = []
    for resolved in resolved_ports:
        artifact = store.latest(resolved.kind, path=resolved.location)
        if artifact is None:
            continue
        staleness = store.is_stale(
            artifact.artifact_id,
            settings=settings if compare_settings else None)
        if staleness.stale:
            notes.append(_note(module, direction, resolved, artifact,
                               staleness))
    return tuple(notes)


def stale_outputs(module: str,
                  settings: Optional[Mapping[str, Any]] = None,
                  *, root: str = "",
                  registry: Optional[Registry] = None
                  ) -> Tuple[StaleNote, ...]:
    """Return the results ``module`` already produced that are out of date.

    The warning a user needs *before* they open a figure or hand a number to
    a collaborator: the measurements in this project were made from a Mask run
    that has since been redone, or with settings that are not the ones now on
    screen.  ``settings`` is compared against the recorded settings hash, so
    editing a material knob marks the existing result stale immediately —
    before the run that would fix it.

    :param module: module key or alias.
    :param settings: the settings currently on screen. Supplying them adds the
        :data:`spacr.artifacts.CAUSE_SETTINGS_CHANGED` cause.
    :param root: explicit project root; otherwise derived from ``settings``.
    :param registry: an open registry instead of the project's own.
    :returns: one note per stale output, in declaration order. Empty when
        there is no registry, which is the answer for a project that has never
        recorded a run.
    """
    spec = _ports.module_ports(module)
    resolved_root = root or _ports.project_root(settings, spec.key)
    store = _registry_for(resolved_root, registry)
    if store is None:
        return ()
    return _walk(spec.key, "output",
                 _ports.declared_outputs(spec.key, root=resolved_root),
                 settings, store, compare_settings=settings is not None)


def stale_inputs(module: str,
                 settings: Optional[Mapping[str, Any]] = None,
                 *, root: str = "",
                 registry: Optional[Registry] = None
                 ) -> Tuple[StaleNote, ...]:
    """Return the inputs ``module`` would read that are already out of date.

    Running on a stale input produces a stale result, so this is the warning
    that saves the twenty minutes rather than explaining them afterwards.
    Settings are deliberately *not* compared here: this module's settings say
    nothing about whether the previous module's output is current.

    :param module: module key or alias.
    :param settings: the settings currently on screen, for the project root.
    :param root: explicit project root; otherwise derived from ``settings``.
    :param registry: an open registry instead of the project's own.
    """
    spec = _ports.module_ports(module)
    resolved_root = root or _ports.project_root(settings, spec.key)
    store = _registry_for(resolved_root, registry)
    if store is None:
        return ()
    return _walk(spec.key, "input",
                 _ports.declared_inputs(spec.key, root=resolved_root),
                 None, store, compare_settings=False)


def staleness_notes(module: str,
                    settings: Optional[Mapping[str, Any]] = None,
                    *, root: str = "",
                    registry: Optional[Registry] = None
                    ) -> Tuple[StaleNote, ...]:
    """Return every stale artifact around ``module``: its inputs and its outputs.

    Inputs first — a stale input explains a stale output, and saying it the
    other way round asks the user to work backwards.

    :param module: module key or alias.
    :param settings: the settings currently on screen.
    :param root: explicit project root.
    :param registry: an open registry instead of the project's own.
    """
    return (stale_inputs(module, settings, root=root, registry=registry)
            + stale_outputs(module, settings, root=root, registry=registry))


# ---------------------------------------------------------------------------
# Continue to the next step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NextStep:
    """A module that can run on what the finished run just produced.

    :param module: the successor's module key.
    :param source: the module that just finished.
    :param root: the project it would run in.
    :param kinds: the kinds it picks up from the finished run.
    :param seed: settings to pre-fill its screen with — the artifact that was
        just produced, resolved through the registry like any other chained
        default. Paths are scalars even for a successor whose key holds a
        list: the receiving end normalises (the chip editor's ``set_value``
        and :func:`spacr.utils.normalize_src_path` both wrap a bare path),
        and guessing the container here would mean keeping a second copy of
        every module's default shape in step with the first.
    :param readiness: :func:`spacr.ports.check_ready`'s verdict.
    :param artifacts: the artifact ids the seed points at.
    """

    module: str
    source: str
    root: str
    kinds: Tuple[str, ...]
    seed: Dict[str, Any]
    readiness: Readiness
    artifacts: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the successor could actually run right now."""
        return bool(self.readiness.ok)

    @property
    def blocked(self) -> str:
        """Why it cannot run, or ``""`` when it can.

        Shown *beside the offer*, not instead of it: a successor that is one
        missing folder away from running is still the thing the user wants,
        and hiding it would leave them wondering where Measure went.
        """
        if self.readiness.ok:
            return ""
        errors = self.readiness.errors
        if not errors:
            return "cannot run here"
        first = errors[0]
        more = len(errors) - 1
        return f"{first.message}{f' (+{more} more)' if more else ''}"

    @property
    def fix(self) -> str:
        """What to do about the blockage, or ``""`` when there is none."""
        errors = self.readiness.errors
        return errors[0].fix if errors else ""


def next_steps(module: str,
               settings: Optional[Mapping[str, Any]] = None,
               *, root: str = "",
               roots: Sequence[str] = (),
               registry: Optional[Registry] = None,
               include_blocked: bool = True) -> Tuple[NextStep, ...]:
    """Return what can run next, pre-filled with what this run just produced.

    Candidates come from :func:`spacr.ports.next_modules` — the modules that
    *require* one of the kinds this one produces — so the list is derived from
    the declared graph rather than a hand-written "after Mask, offer Measure".
    Each is resolved against the registry for its settings, then run through
    :func:`spacr.ports.check_ready`, so an offer either works or says why not.

    :param module: the module that just finished; key or alias.
    :param settings: the settings it ran with, for the project root.
    :param root: explicit project root.
    :param roots: further project roots for the successor's inputs.
    :param registry: an open registry instead of the project's own.
    :param include_blocked: keep successors that cannot run, carrying their
        blocking reason. False drops them entirely.
    :returns: one :class:`NextStep` per successor, ready ones first, then in
        module order.
    :raises spacr.ports.UnknownModule: when ``module`` declares no ports.
    """
    spec = _ports.module_ports(module)
    resolved_root = root or _ports.project_root(settings, spec.key)
    produced = {port.kind for port in spec.produces}
    # Opened once for the whole answer: every successor is checked against the
    # same project, and each open runs the registry's schema DDL.
    store = _registry_for(resolved_root, registry)
    steps: List[NextStep] = []
    for candidate in _ports.next_modules(spec.key):
        successor = _ports.module_ports(candidate)
        kinds = tuple(sorted({port.kind for port in successor.consumes
                              if port.kind in produced and port.required}))
        seed: Dict[str, Any] = {source_key(candidate): ""}
        inputs = chained_inputs(candidate, seed, root=resolved_root,
                                roots=roots, registry=registry,
                                check_staleness=False)
        for chained in inputs:
            seed.setdefault(chained.setting, "")
            if is_empty_path(seed[chained.setting]):
                seed[chained.setting] = chained.value
        if is_empty_path(seed.get(source_key(candidate))) and resolved_root:
            # No registry row yet — the successor still runs in the project
            # the finished module ran in, and saying so is better than
            # handing over an empty screen.
            seed[source_key(candidate)] = resolved_root
        readiness = _ports.check_ready(candidate, seed, registry=store)
        step = NextStep(
            module=candidate, source=spec.key, root=resolved_root,
            kinds=kinds, seed=seed, readiness=readiness,
            artifacts=tuple(c.artifact.artifact_id for c in inputs))
        if step.ok or include_blocked:
            steps.append(step)
    steps.sort(key=lambda s: (not s.ok, s.module))
    return tuple(steps)


# ---------------------------------------------------------------------------
# Layout-aware drops
# ---------------------------------------------------------------------------
#
# A dropped folder and an auto-chained one have to arrive at the same answer.
# Two answers to "where is the database" is how a screen and the run it
# launches come to disagree, so the drop path does not re-derive anything: it
# asks the registry through :func:`chained_inputs` exactly as auto-chaining
# does, and only when the registry has nothing does it fall back to the
# declared layout in :data:`spacr.ports.PORTS`.
#
# The fallback is the difference between the two, and it is additive: where
# auto-chaining leaves a field empty because no run was ever registered, a
# drop still fills it from the folder the user just pointed at. Where the
# registry *does* have a row, both produce the same string.

#: Suffixes a SQLite measurements database is written with.
DB_SUFFIXES: Tuple[str, ...] = (".db", ".sqlite", ".sqlite3")

#: Suffixes a result table is written with, in the order a picker offers them.
TABLE_SUFFIXES: Tuple[str, ...] = (".csv", ".tsv", ".parquet")

#: How far above a dropped path the project root may sit. Four covers the
#: deepest declared layout, ``data/<plate>/<class>_png/<file>``.
MAX_CLIMB = 4

#: How many children of a dropped folder are examined when looking for the
#: projects inside it. A drop happens while the user is holding the mouse
#: button down, so the search is bounded rather than exhaustive: somebody who
#: drops a folder of two thousand plates is answering a different question.
MAX_CHILDREN = 200

#: Where a drop's answer came from.
FROM_REGISTRY = "registry"
FROM_LAYOUT = "layout"

#: What a screen that takes a whole project resolves to. Deliberately *not* a
#: member of :data:`spacr.ports.ALL_KINDS`: the project folder is not an
#: artifact any module produces, it is the thing artifacts live in, and adding
#: it to the port vocabulary would put it in the module graph.
PROJECT = "project"

_LAYOUT_CACHE: Dict[str, Any] = {}


def _first_component(relative: str) -> str:
    """Return the leading path component of ``relative``, or ``""``."""
    head = os.path.normpath(relative).split(os.sep)[0]
    if head in ("", ".", os.sep) or any(ch in head for ch in "*?["):
        return ""
    return head


def layout_directories() -> Tuple[str, ...]:
    """Return the folder names spaCR's project layout uses, sorted.

    Read off :data:`spacr.ports.PORTS` rather than typed out — ``merged``,
    ``measurements``, ``masks``, ``data``, ``model``, ``results``,
    ``settings``, ``orig``, ``consolidated`` all come from a declaration
    somebody already wrote — so a plugin that declares a port makes its own
    folder part of the layout without editing a list here.

    Cached against the size of the registry, so a late
    :func:`spacr.ports.register_module_ports` is picked up.
    """
    if _LAYOUT_CACHE.get("size") == len(_ports.PORTS):
        return _LAYOUT_CACHE["dirs"]
    names: set = set()
    for spec in _ports.PORTS.values():
        for port in spec.consumes + spec.produces:
            if port.path:
                if os.sep in os.path.normpath(port.path):
                    names.add(_first_component(port.path))
                elif port.pattern or not os.path.splitext(port.path)[1]:
                    names.add(_first_component(port.path))
            for alternative in port.pattern.split("|"):
                if "/" in alternative:
                    names.add(_first_component(alternative))
    dirs = tuple(sorted(n for n in names if n))
    _LAYOUT_CACHE.update(size=len(_ports.PORTS), dirs=dirs)
    return dirs


def project_root_of(path: Any, *, max_climb: int = MAX_CLIMB) -> str:
    """Return the project root a dropped path belongs to.

    The layout is walked *upwards*: ``<root>/measurements/measurements.db``,
    ``<root>/merged``, ``<root>/data/plate1/cell_png`` and ``<root>`` itself
    all answer ``<root>``, because ``measurements``, ``merged`` and ``data``
    are declared folders (:func:`layout_directories`) and nothing else on the
    way up is.

    The highest declared folder within ``max_climb`` wins, so a drop deep
    inside ``data/`` still lands on the project rather than on a crop folder.

    :param path: the dropped file or folder.
    :param max_climb: how many levels above the drop to consider.
    :returns: an absolute path. Never raises: a path that is nowhere near a
        project answers with its own folder, which is what a direct drop
        wants anyway.
    """
    if path is None:
        return ""
    current = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isfile(current) or os.path.splitext(current)[1]:
        current = os.path.dirname(current)
    known = set(layout_directories())
    root = current
    for _ in range(max_climb + 1):
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        if os.path.basename(current) in known:
            root = parent
        current = parent
    return root


def ports_for_kinds(kinds: Sequence[str]) -> Tuple[Port, ...]:
    """Return the canonical declaration of where each kind lives.

    A screen that is not a pipeline module — the table explorers, the
    viewers — still says what it wants in the shared vocabulary, and this is
    what turns that word into a path. The declaration is looked up in
    :data:`spacr.ports.PORTS`: a *produced* port first, because the module
    that writes a kind is the one that knows where it goes, and a consumed
    port only when nothing produces it.

    :param kinds: vocabulary terms such as :data:`spacr.ports.MEASUREMENTS_DB`.
    :returns: one :class:`spacr.ports.Port` per kind that is declared
        anywhere, in the order asked for. An undeclared kind is skipped
        rather than guessed at.
    """
    produced: Dict[str, Port] = {}
    consumed: Dict[str, Port] = {}
    for spec in _ports.PORTS.values():
        for port in spec.produces:
            produced.setdefault(port.kind, port)
        for port in spec.consumes:
            consumed.setdefault(port.kind, port)
    found: List[Port] = []
    for kind in kinds:
        port = produced.get(kind) or consumed.get(kind)
        if port is not None:
            found.append(port)
    return tuple(found)


def db_candidates(root: str) -> Tuple[str, ...]:
    """Return every SQLite database in a project, the declared one first.

    The declared location comes from the :data:`spacr.ports.MEASUREMENTS_DB`
    port; the rest is a shallow listing of the root and of the folder that
    port names. Two databases in one project is not an error and not a thing
    to guess about — it is a question, and this is the list to ask it with.
    """
    if not root or not os.path.isdir(root):
        return ()
    declared = ports_for_kinds((_ports.MEASUREMENTS_DB,))
    found: List[str] = []
    folders: List[str] = [root]
    for port in declared:
        target = os.path.join(root, port.path) if port.path else root
        if os.path.isfile(target):
            found.append(target)
        holder = os.path.dirname(target)
        if os.path.isdir(holder) and holder not in folders:
            folders.append(holder)
    for folder in folders:
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in entries:
            candidate = os.path.join(folder, name)
            if (name.lower().endswith(DB_SUFFIXES)
                    and os.path.isfile(candidate)
                    and candidate not in found):
                found.append(candidate)
    return tuple(found)


def result_tables(root: str) -> Tuple[str, ...]:
    """Return the result tables a project has written, sorted.

    The folders searched are the ones the result-bearing ports declare —
    ``results/`` and ``settings/`` today — one level deep, so a drop on a
    screen that reads "a table or a CSV" can offer the CSVs beside the
    database tables instead of making the user go and find them.
    """
    if not root or not os.path.isdir(root):
        return ()
    kinds = (_ports.REGRESSION_RESULTS, _ports.EMBEDDING, _ports.SETTINGS_CSV)
    folders: List[str] = []
    for port in ports_for_kinds(kinds):
        target = os.path.join(root, port.path) if port.path else root
        holder = target if not os.path.splitext(target)[1] else os.path.dirname(target)
        if os.path.isdir(holder) and holder not in folders:
            folders.append(holder)
    found: List[str] = []
    for folder in folders:
        for base, _dirs, files in os.walk(folder):
            if os.path.relpath(base, folder).count(os.sep) >= 1:
                _dirs[:] = []
            for name in files:
                if name.lower().endswith(TABLE_SUFFIXES):
                    found.append(os.path.join(base, name))
    return tuple(sorted(dict.fromkeys(found)))


@dataclass(frozen=True)
class DropTarget:
    """One settings key a drop can fill, and what it resolved to.

    :param module: the screen the drop landed on.
    :param setting: the settings key it fills.
    :param role: the port's role.
    :param kind: the :mod:`spacr.ports` kind.
    :param value: what the settings key should become — the project root for
        a :data:`ROOT` binding, the artifact itself for a :data:`PATH` one.
    :param location: the artifact's own path, always. This is what an
        interface shows the user: "it resolved to *this*".
    :param source: :data:`FROM_REGISTRY` when the answer came from a
        recorded run — the same answer auto-chaining gives — or
        :data:`FROM_LAYOUT` when it came from the declared folder layout.
    :param required: whether the screen needs this input.
    :param paths: the individual files the port's pattern matched, for a
        screen that wants one file rather than the folder holding them.
    """

    module: str
    setting: str
    role: str
    kind: str
    value: Any
    location: str
    source: str
    required: bool = True
    paths: Tuple[str, ...] = ()

    def describe(self) -> str:
        """One line naming what was found and where."""
        return f"{self.kind} → {self.location} (from the {self.source})"


@dataclass(frozen=True)
class DropChoice:
    """A question a drop cannot answer on its own.

    Two databases in a folder, two projects under the folder that was
    dropped, two tables in the database — each has a right answer and none of
    them is "the first one". :attr:`options` is what to offer.

    :param question: the sentence to put above the list.
    :param kind: the vocabulary term the options are candidates for.
    :param options: the candidates, in the order to offer them.
    :param setting: the settings key the answer fills, when there is one.
    """

    question: str
    kind: str
    options: Tuple[str, ...]
    setting: str = ""


@dataclass(frozen=True)
class DropResolution:
    """What a dropped path means to one screen.

    :param module: the screen key.
    :param dropped: the path the user dropped, absolute.
    :param root: the project root it resolved to.
    :param targets: the inputs that were found.
    :param choices: the questions that have to be asked first.
    :param problems: :class:`spacr.validate.Problem` for every input that is
        missing — the same sentences :func:`spacr.ports.check_ready` writes,
        because they come from it.
    """

    module: str
    dropped: str
    root: str
    targets: Tuple[DropTarget, ...] = ()
    choices: Tuple[DropChoice, ...] = ()
    problems: Tuple[Any, ...] = ()

    def __bool__(self) -> bool:
        """True when the drop resolved to something usable."""
        return self.ok

    @property
    def ok(self) -> bool:
        """True when every required input was found."""
        return bool(self.targets) and not any(
            p.is_error for p in self.problems)

    @property
    def ambiguous(self) -> bool:
        """True when the drop has to be asked about rather than applied."""
        return bool(self.choices)

    def target_for(self, kind: str) -> Optional[DropTarget]:
        """Return the resolved target of ``kind``, or None."""
        for target in self.targets:
            if target.kind == kind:
                return target
        return None

    @property
    def reason(self) -> str:
        """One human-readable line: what it resolved to, or why it did not."""
        if self.choices:
            choice = self.choices[0]
            return f"{choice.question} ({len(choice.options)} candidates)"
        if self.targets:
            return "; ".join(t.describe() for t in self.targets)
        errors = [p for p in self.problems if p.is_error]
        if errors:
            return f"{errors[0].message}. {errors[0].fix}"
        return f"nothing this module reads was found in {self.root}"


def looks_laid_out(folder: str) -> bool:
    """True when ``folder`` holds any of spaCR's declared layout folders.

    The cheap structural answer to "is this a project?", nine ``stat`` calls
    against :func:`layout_directories`. :func:`spacr.projects.looks_like_project`
    is the thorough one and reads the registry and every module's outputs;
    a drop happens while the mouse button is still down, so this is the one
    that runs there.
    """
    if not folder or not os.path.isdir(folder):
        return False
    return any(os.path.isdir(os.path.join(folder, name))
               for name in layout_directories())


def satisfies(root: str, ports: Sequence[Port]) -> bool:
    """True when ``root`` holds everything ``ports`` requires.

    With no ports the question is "is this a project at all?", which is what a
    screen that takes a whole project — the pipeline graph, the QC dashboard —
    is asking.
    """
    if not ports:
        return looks_laid_out(root)
    return all(_ports.resolve_port(port, root).exists
               for port in ports if port.required)


def _problems_for(module: str, ports: Sequence[Port], root: str,
                  registry: Optional[Registry]) -> Tuple[Any, ...]:
    """Say why a drop found nothing, in :func:`check_ready`'s own words."""
    if not ports:
        from .validate import ERROR, Problem
        return (Problem(
            ERROR, PROJECT,
            f"{root} is not a spaCR project folder",
            "Drop the plate folder itself — the one holding "
            f"{', '.join(layout_directories()[:4])} and the rest of the "
            "layout — or a folder containing several of them."),)
    try:
        readiness = _ports.check_ready(module, root=root, registry=registry)
    except _ports.UnknownModule:
        pass
    else:
        return readiness.problems
    problems: List[Any] = []
    for port in ports:
        problems.extend(_ports.port_problems(port, root))
    return tuple(problems)


def _sub_projects(folder: str, ports: Sequence[Port]) -> Tuple[str, ...]:
    """Return the immediate children of ``folder`` that satisfy ``ports``.

    Dropping the folder that holds a screen's plates is a normal thing to do
    and it has no single answer, so it becomes a question rather than a guess.

    A child named by the layout — ``masks``, ``merged``, ``results`` — is
    never a sub-project. Without that exclusion, a plate whose raw images had
    been cleaned away answered "did you mean ``masks/``?", because a folder of
    label TIFFs does satisfy a raw-image port when you only look at file
    extensions.
    """
    if not folder or not os.path.isdir(folder):
        return ()
    known = set(layout_directories())
    found: List[str] = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return ()
    for name in entries[:MAX_CHILDREN]:
        child = os.path.join(folder, name)
        if name in known or not os.path.isdir(child):
            continue
        if satisfies(child, ports):
            found.append(child)
    return tuple(found)


def resolve_drop(module: str,
                 dropped: Any,
                 *,
                 kinds: Sequence[str] = (),
                 form: str = PATH,
                 settings: Optional[Mapping[str, Any]] = None,
                 registry: Optional[Registry] = None,
                 max_climb: int = MAX_CLIMB) -> DropResolution:
    """Work out what a dropped path means to ``module``.

    The whole point of the function is that it is *the same* resolution
    auto-chaining performs. For every input the module declares, the registry
    is asked first — :meth:`spacr.artifacts.Registry.latest` for that kind in
    that project, exactly as :func:`chained_inputs` asks it — so a drop and an
    auto-chain fill the field with the same string. Only when no run was ever
    registered does the declared layout in :data:`spacr.ports.PORTS` answer
    instead, and then it answers with the folder the ports say it is in.

    Ambiguity is returned, never guessed:

    * the dropped folder holds several projects → :class:`DropChoice`;
    * the project holds several databases → :class:`DropChoice`;
    * nothing satisfies the module → :attr:`DropResolution.problems`, which
      is :func:`spacr.ports.check_ready`'s own list of sentences.

    :param module: the screen key. When it declares ports those are used;
        otherwise ``kinds`` says what it wants.
    :param dropped: the path the user dropped.
    :param kinds: vocabulary terms, for a screen with no port declaration.
    :param form: :data:`ROOT` or :data:`PATH` — what a ``kinds``-driven screen
        wants in its field. A declared module's own bindings always win.
    :param settings: the settings dict being edited, for its current values.
    :param registry: an open registry to ask instead of each project's own.
    :param max_climb: how far above the drop the project root may sit.
    :returns: a :class:`DropResolution`.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(dropped)))
    key = _canonical(module)
    try:
        spec = _ports.module_ports(key)
        ports = tuple(spec.consumes)
        key = spec.key
        declared = True
    except _ports.UnknownModule:
        ports = ports_for_kinds(kinds)
        declared = False
    if not ports:
        ports = ports_for_kinds(kinds)

    climbed = project_root_of(path, max_climb=max_climb)
    direct = path if os.path.isdir(path) else os.path.dirname(path)
    candidates = [r for r in (climbed, direct, os.path.dirname(direct)) if r]
    ordered: List[str] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)

    root = ordered[0] if ordered else direct
    for candidate in ordered:
        if satisfies(candidate, ports):
            root = candidate
            break

    choices: List[DropChoice] = []
    satisfied = satisfies(root, ports)
    if not satisfied:
        children = _sub_projects(direct, ports)
        if len(children) == 1:
            root = children[0]
            satisfied = True
        elif len(children) > 1:
            choices.append(DropChoice(
                question=f"{len(children)} projects under "
                         f"{os.path.basename(direct)} can be used here — "
                         f"which one?",
                kind=ports[0].kind if ports else "",
                options=children,
                setting=(binding_for(key, ports[0]).setting if ports else "")))

    stores: Dict[str, Optional[Registry]] = {}
    targets: List[DropTarget] = []
    if not ports and satisfied:
        # A screen that takes the project itself. There is no port to resolve
        # and nothing to look up: the answer is the folder the layout walk
        # arrived at, which is the point of having walked it.
        targets.append(DropTarget(
            module=key, setting=source_key(key), role=PROJECT, kind=PROJECT,
            value=root, location=root, source=FROM_LAYOUT))
    filled: set = set()
    for port in ports:
        # A port that is not declared by a module has no settings key of its
        # own; its role stands in, so a screen asking for two kinds gets two
        # answers rather than two ports fighting over ``src``.
        binding = (binding_for(key, port) if declared
                   else Binding(key, port.role, port.role, form))
        if binding.setting in filled:
            # The key already has its answer. Classify declares both a
            # measurements database and an optional ``data/**/*_png`` crop
            # folder, and *both* bind to ``src`` — so resolving the second
            # would recursively glob a folder of a hundred thousand crops to
            # arrive at the string already in hand. A drop happens with the
            # mouse button down; this is the difference between one
            # millisecond and forty.
            continue
        current = None if settings is None else settings.get(binding.setting)
        if root not in stores:
            stores[root] = _registry_for(root, registry)
        store = stores[root]
        artifact = None if store is None else store.latest(port.kind,
                                                           project=root)
        if artifact is not None:
            targets.append(DropTarget(
                module=key, setting=binding.setting, role=port.role,
                kind=port.kind,
                value=_value_for(artifact, port, binding, current),
                location=artifact.path, source=FROM_REGISTRY,
                required=port.required))
            filled.add(binding.setting)
            continue
        resolved = _ports.resolve_port(port, root)
        if not resolved.exists:
            continue
        # ``target`` and not ``paths[0]``: the port's declared location is the
        # artifact, whether that is one file (``measurements/measurements.db``)
        # or the folder a pattern selects inside (``merged/*.npy``). Naming
        # the first matching file would make a re-drop of the same folder
        # resolve differently as soon as another field was written.
        location = resolved.target
        value = root if binding.form == ROOT else location
        if isinstance(current, (list, tuple)):
            value = [value]
        targets.append(DropTarget(
            module=key, setting=binding.setting, role=port.role,
            kind=port.kind, value=value, location=location,
            source=FROM_LAYOUT, required=port.required,
            paths=resolved.paths))
        filled.add(binding.setting)

    # A database is the one artifact a project can legitimately hold two of.
    # Picking the first would be exactly the silent wrong answer this is here
    # to avoid.
    if any(t.kind == _ports.MEASUREMENTS_DB for t in targets):
        available = db_candidates(root)
        if len(available) > 1:
            chosen = next(t for t in targets
                          if t.kind == _ports.MEASUREMENTS_DB)
            choices.append(DropChoice(
                question=f"{os.path.basename(root)} holds "
                         f"{len(available)} databases — which one?",
                kind=_ports.MEASUREMENTS_DB, options=available,
                setting=chosen.setting))

    problems: Tuple[Any, ...] = ()
    if not targets and not choices:
        problems = _problems_for(key if declared else "", ports, root,
                                 registry)
    return DropResolution(module=key, dropped=path, root=root,
                          targets=tuple(targets), choices=tuple(choices),
                          problems=problems)
