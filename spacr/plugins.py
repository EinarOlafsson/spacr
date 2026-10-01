"""Versioned extension SDK for third-party spaCR plugins.

Plugins are ordinary Python distributions exposing one entry point in the
``spacr.plugins`` group.  The entry point may resolve to a
:class:`SpacrPlugin`, a mapping accepted by :func:`plugin_from_mapping`, or a
zero-argument factory returning either.  Discovery is lazy, deterministic and
failure-isolated: one malformed plugin is recorded in :func:`diagnostics`
without preventing spaCR or the remaining plugins from loading.

For editable/local development, ``SPACR_PLUGIN_MODULES`` may contain a
comma-separated list of ``module`` or ``module:attribute`` references.
Installed plugins should always use package entry points instead.

Plugins and assay recipes can also be installed from a catalogue, a JSON
file listing each entry's version, author and licence. A catalogue plugin is
installed into its own folder under ``~/.spacr/plugins`` (or
``SPACR_PLUGIN_HOME``), together with the libraries it asks for, so it never
replaces or upgrades a package spaCR itself uses; it is discovered like any
other plugin until it is uninstalled.

Setting ``SPACR_DISABLE_PLUGINS`` to ``1``, ``true``, ``yes`` or ``on``
(case-insensitively) skips discovery entirely, so no plugin loads from either
source.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_CATALOGUE_ENV = "SPACR_PLUGIN_CATALOGUE"
_PLUGIN_HOME_ENV = "SPACR_PLUGIN_HOME"
_CATALOGUE_FILE = "catalogue.json"
_INSTALLED_FILE = "installed.json"
_CATALOGUE_KINDS = ("plugin", "recipe")

__all__ = [
    "PLUGIN_API_VERSION",
    "PLUGIN_ENTRY_POINT_GROUP",
    "AppContribution",
    "ModelProviderContribution",
    "PluginDiagnostic",
    "ReportContext",
    "ReportSectionContribution",
    "SpacrPlugin",
    "diagnostics",
    "discover_plugins",
    "get_app",
    "load_object",
    "model_providers",
    "plugin_apps",
    "plugin_from_mapping",
    "record_diagnostic",
    "reload_plugins",
    "report_sections",
]

PLUGIN_API_VERSION = "1.0"
PLUGIN_ENTRY_POINT_GROUP = "spacr.plugins"
PLUGIN_MODULES_ENV = "SPACR_PLUGIN_MODULES"
DISABLE_PLUGINS_ENV = "SPACR_DISABLE_PLUGINS"

LOG = logging.getLogger(__name__)
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_SECTIONS = frozenset({"core", "data", "models", "results", "toxo"})
_STAGES = frozenset({"alpha", "beta", "stable"})
_KINDS = frozenset({"assay", "importer", "analysis", "utility"})
_CALL_STYLES = frozenset({"settings", "folder"})


@dataclass(frozen=True)
class AppContribution:
    """One runnable GUI/headless application contributed by a plugin.

    Discovery rejects the contribution with :exc:`ValueError` when ``section``,
    ``stage``, ``kind`` or ``call_style`` falls outside the fixed vocabulary
    listed below, when ``key`` does not match ``^[a-z][a-z0-9_]{1,63}$``, or
    when any non-empty ``module:callable`` reference is malformed.

    :ivar key: identifier the CLI accepts and the GUI registers under; must
        match ``^[a-z][a-z0-9_]{1,63}$``. Reusing another plugin's key fails
        that plugin's load; colliding with a built-in app skips the
        contribution and records a diagnostic.
    :ivar name: human-readable title shown in the sidebar and screen header;
        cannot be blank.
    :ivar description: one-line blurb used as the app intro and the
        ``spacr-run --list`` summary; cannot be blank.
    :ivar entrypoint: ``"module:callable"`` reference to the callable that
        does the work.
    :ivar defaults: ``"module:callable"`` reference to a helper returning the
        settings dictionary; called with ``{}`` and retried with no argument.
    :ivar section: sidebar group; one of ``"core"``, ``"data"``, ``"models"``,
        ``"results"`` or ``"toxo"``.
    :ivar stage: maturity annotation; one of ``"alpha"``, ``"beta"`` or
        ``"stable"``.
    :ivar kind: what the app is; one of ``"assay"``, ``"importer"``,
        ``"analysis"`` or ``"utility"``.
    :ivar categories: settings-screen tabs, mapping a tab name to the setting
        keys it holds; empty means the generic ungrouped layout.
    :ivar tooltips: hover text per setting key.
    :ivar labels: display label per setting key, overriding the generated one.
    :ivar docs_url: address the settings screen's API link opens.
    :ivar aliases: extra names the CLI and :mod:`spacr.validate` resolve to
        :attr:`key`.
    :ivar validator: optional ``"module:callable"`` reference to a callable
        taking the settings dict and returning ``spacr.validate.Problem``
        objects or equivalent mappings.
    :ivar screen_factory: optional ``"module:callable"`` reference to a
        factory returning a ``QWidget``, replacing the generic settings
        screen; it is always invoked as ``factory(app_key=...)`` and so must
        accept that keyword.
    :ivar drop_handler: optional ``"module:callable"`` reference to a
        ``spacr.qt.dnd_handlers.DropHandler`` subclass.
    :ivar icon: absolute image path, or the name of a spaCR semantic icon;
        empty falls back to the puzzle-piece icon.
    :ivar requires: settings the user must supply, phrased for a human.
    :ivar writes: what the app leaves on disk.
    :ivar call_style: ``"settings"`` for ``fn(settings_dict)``; ``"folder"``
        for a callable taking a bare path.
    """

    key: str
    name: str
    description: str
    entrypoint: str
    defaults: str
    section: str = "results"
    stage: str = "alpha"
    kind: str = "analysis"
    categories: Mapping[str, Sequence[str]] = field(default_factory=dict)
    tooltips: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    docs_url: str = ""
    aliases: Tuple[str, ...] = ()
    validator: str = ""
    screen_factory: str = ""
    drop_handler: str = ""
    icon: str = ""
    requires: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    call_style: str = "settings"


@dataclass(frozen=True)
class ModelProviderContribution:
    """Immutable record naming a plugin's model-zoo provider.

    :param key: identifier for the provider; must match
        ``^[a-z][a-z0-9_]{1,63}$`` and be unique across all loaded plugins.
    :param provider: ``"module:callable"`` reference string -- not the callable
        itself -- resolved at catalogue time to a zero-argument callable
        returning model-zoo entries or entry mappings.
    """

    key: str
    provider: str


@dataclass(frozen=True)
class ReportSectionContribution:
    """Immutable record naming a builder that adds one report section.

    :func:`spacr.report.collect_report` resolves and calls the builder, and
    substitutes a visible problem section if it fails.

    :param key: stable section identifier; plugin validation requires
        ``^[a-z][a-z0-9_]{1,63}$`` and discovery rejects duplicate
        contribution keys.
    :param title: fallback section heading used when the builder returns no
        title and when the builder fails; cannot be blank.
    :param builder: ``"module:callable"`` reference resolved at report
        collection to a callable taking :class:`ReportContext` and returning
        a :class:`spacr.report.Section`.
    :param after: existing section key after which this section is inserted;
        an unmatched key appends it to the report.
    """

    key: str
    title: str
    builder: str
    after: str = "statistics"


@dataclass(frozen=True)
class SpacrPlugin:
    """Validated plugin manifest returned by a ``spacr.plugins`` entry point.

    :param name: human-readable plugin name used in diagnostics and discovery
        output.
    :param version: version of the plugin distribution, reported to users
        without being interpreted by spaCR.
    :param api_version: plugin SDK version the manifest targets; its major
        version must match :data:`PLUGIN_API_VERSION`.
    :param apps: runnable applications the plugin adds to the GUI and headless
        registry.
    :param model_providers: providers that extend the model-zoo catalogue.
    :param report_sections: builders that insert plugin-owned sections into
        generated reports.
    :param translations: locale-to-message mappings that translate the
        plugin's own visible strings.
    """

    name: str
    version: str
    api_version: str = PLUGIN_API_VERSION
    apps: Tuple[AppContribution, ...] = ()
    model_providers: Tuple[ModelProviderContribution, ...] = ()
    report_sections: Tuple[ReportSectionContribution, ...] = ()
    translations: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginDiagnostic:
    """One discovery or contribution error visible to users and logs.

    :param plugin: entry-point or manifest name identifying the plugin that
        could not be loaded.
    :param severity: diagnostic level, such as ``"error"`` or ``"warning"``.
    :param message: concise user-facing account of the failed operation.
    :param exception: captured exception text with the technical cause; empty
        when no exception accompanied the diagnostic.
    """

    plugin: str
    severity: str
    message: str
    exception: str = ""


@dataclass(frozen=True)
class ReportContext:
    """Read-only inputs passed to plugin report-section builders.

    :param src: source folder or object from which the core report is built.
    :param artifacts: named core report artifacts available for reuse by the
        plugin section.
    :param runs: immutable sequence of recorded run summaries associated with
        the report source.
    :param options: report-generation options supplied by the caller.
    """

    src: Any
    artifacts: Mapping[str, Any]
    runs: Tuple[Mapping[str, Any], ...]
    options: Mapping[str, Any]


@dataclass
class _Registry:
    """Everything the loaded plugins contribute, in one immutable snapshot.

    ONE OBJECT REPLACED WHOLESALE rather than several mutable collections
    kept in step. A plugin that fails half way through contributing would
    otherwise leave the registry holding its apps and not its models, and
    nothing would say so; building a new `_Registry` and swapping it means
    a partial load is discarded rather than published.

    Tuples for the ordered contributions so plugin order is reproducible,
    and a dict for `apps` because they are looked up by key.
    """

    plugins: Tuple[SpacrPlugin, ...] = ()
    apps: Dict[str, AppContribution] = field(default_factory=dict)
    models: Tuple[Tuple[str, ModelProviderContribution], ...] = ()
    reports: Tuple[Tuple[str, ReportSectionContribution], ...] = ()
    diagnostics: List[PluginDiagnostic] = field(default_factory=list)


_LOCK = threading.RLock()
_REGISTRY: Optional[_Registry] = None


def load_object(reference: str) -> Any:
    """Import and return ``module:attribute`` (nested attributes supported).

    :param reference: string of the form ``"package.module:attribute"``; the
        attribute part may be dotted to reach nested objects. Anything else
        raises ``ValueError``.
    """
    if not isinstance(reference, str) or not _REF_RE.match(reference):
        raise ValueError(
            f"invalid object reference {reference!r}; expected 'package.module:attribute'"
        )
    module_name, path = reference.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _tuple_strings(value: Any, field_name: str) -> Tuple[str, ...]:
    """Normalize an optional string sequence into a nonblank string tuple."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} cannot contain blank values")
    return result


def _mapping_of_strings(value: Any, field_name: str) -> Dict[str, str]:
    """Normalize an optional mapping by converting every key and value to text."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _app_from_mapping(value: Any) -> AppContribution:
    """Coerce and validate one application contribution."""
    if isinstance(value, AppContribution):
        app = value
    elif isinstance(value, Mapping):
        data = dict(value)
        data["aliases"] = _tuple_strings(data.get("aliases"), "aliases")
        data["requires"] = _tuple_strings(data.get("requires"), "requires")
        data["writes"] = _tuple_strings(data.get("writes"), "writes")
        data["tooltips"] = _mapping_of_strings(data.get("tooltips"), "tooltips")
        data["labels"] = _mapping_of_strings(data.get("labels"), "labels")
        categories = data.get("categories") or {}
        if not isinstance(categories, Mapping):
            raise TypeError("categories must be a mapping of tab names to setting keys")
        data["categories"] = {
            str(name): _tuple_strings(keys, f"categories[{name!r}]")
            for name, keys in categories.items()
        }
        app = AppContribution(**data)
    else:
        raise TypeError("apps entries must be AppContribution objects or mappings")
    if not _KEY_RE.match(app.key):
        raise ValueError(f"invalid app key {app.key!r}")
    if not app.name.strip() or not app.description.strip():
        raise ValueError(f"plugin app {app.key!r} needs a name and description")
    if app.section not in _SECTIONS:
        raise ValueError(f"plugin app {app.key!r} has unknown section {app.section!r}")
    if app.stage not in _STAGES:
        raise ValueError(f"plugin app {app.key!r} has unknown stage {app.stage!r}")
    if app.kind not in _KINDS:
        raise ValueError(f"plugin app {app.key!r} has unknown kind {app.kind!r}")
    if app.call_style not in _CALL_STYLES:
        raise ValueError(f"plugin app {app.key!r} has invalid call_style")
    for label, reference in (
        ("entrypoint", app.entrypoint),
        ("defaults", app.defaults),
        ("validator", app.validator),
        ("screen_factory", app.screen_factory),
        ("drop_handler", app.drop_handler),
    ):
        if reference and not _REF_RE.match(reference):
            raise ValueError(f"plugin app {app.key!r} has invalid {label} reference")
    return app


def _model_from_mapping(value: Any) -> ModelProviderContribution:
    """Coerce and validate one model-provider contribution."""
    if isinstance(value, ModelProviderContribution):
        contribution = value
    elif isinstance(value, Mapping):
        contribution = ModelProviderContribution(**dict(value))
    else:
        raise TypeError("model_providers entries must be contributions or mappings")
    if not _KEY_RE.match(contribution.key) or not _REF_RE.match(contribution.provider):
        raise ValueError("model provider needs a valid key and module:callable reference")
    return contribution


def _report_from_mapping(value: Any) -> ReportSectionContribution:
    """Coerce and validate one report-section contribution."""
    if isinstance(value, ReportSectionContribution):
        contribution = value
    elif isinstance(value, Mapping):
        contribution = ReportSectionContribution(**dict(value))
    else:
        raise TypeError("report_sections entries must be contributions or mappings")
    if not _KEY_RE.match(contribution.key) or not contribution.title.strip():
        raise ValueError("report section needs a valid key and title")
    if not _REF_RE.match(contribution.builder):
        raise ValueError("report section builder must be a module:callable reference")
    return contribution


def plugin_from_mapping(value: Mapping[str, Any]) -> SpacrPlugin:
    """Validate a mapping and return its immutable :class:`SpacrPlugin`.

    :param value: manifest mapping whose keys are the :class:`SpacrPlugin`
        fields; ``apps``, ``model_providers`` and ``report_sections`` may hold
        mappings or contribution objects, and ``translations`` must map
        language codes to string mappings. A non-mapping raises ``TypeError``.
    """
    if not isinstance(value, Mapping):
        raise TypeError("plugin manifest must be a mapping")
    data = dict(value)
    data["apps"] = tuple(_app_from_mapping(item) for item in data.get("apps", ()))
    data["model_providers"] = tuple(
        _model_from_mapping(item) for item in data.get("model_providers", ())
    )
    data["report_sections"] = tuple(
        _report_from_mapping(item) for item in data.get("report_sections", ())
    )
    translations = data.get("translations") or {}
    if not isinstance(translations, Mapping):
        raise TypeError("translations must map language codes to message mappings")
    data["translations"] = {
        str(language): _mapping_of_strings(messages, f"translations[{language!r}]")
        for language, messages in translations.items()
    }
    plugin = SpacrPlugin(**data)
    _validate_plugin(plugin)
    return plugin


def _validate_plugin(plugin: SpacrPlugin) -> None:
    """Validate plugin metadata, contribution shapes, and unique local keys."""
    if not plugin.name.strip() or not plugin.version.strip():
        raise ValueError("plugin name and version are required")
    if plugin.api_version.split(".", 1)[0] != PLUGIN_API_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"plugin requires SDK {plugin.api_version}; spaCR provides {PLUGIN_API_VERSION}"
        )
    groups = (
        ("app", tuple(_app_from_mapping(item) for item in plugin.apps)),
        ("model provider", tuple(
            _model_from_mapping(item) for item in plugin.model_providers
        )),
        ("report section", tuple(
            _report_from_mapping(item) for item in plugin.report_sections
        )),
    )
    for label, items in groups:
        keys = [item.key for item in items]
        duplicate = next((key for key in keys if keys.count(key) > 1), "")
        if duplicate:
            raise ValueError(
                f"plugin {plugin.name!r} repeats {label} key {duplicate!r}"
            )
    if not isinstance(plugin.translations, Mapping):
        raise TypeError("plugin translations must be a mapping")
    for language, messages in plugin.translations.items():
        _mapping_of_strings(messages, f"translations[{language!r}]")


def _coerce_plugin(value: Any) -> SpacrPlugin:
    """Resolve an entry-point value or factory into a validated plugin."""
    if callable(value) and not isinstance(value, type):
        value = value()
    if isinstance(value, SpacrPlugin):
        _validate_plugin(value)
        return value
    if isinstance(value, Mapping):
        return plugin_from_mapping(value)
    raise TypeError("entry point must expose SpacrPlugin, a manifest mapping, or a factory")


def _installed_sources() -> Iterable[Tuple[str, Callable[[], Any]]]:
    """Yield named plugin loaders from installed entry points and the environment."""
    from importlib import metadata

    try:
        discovered = metadata.entry_points()
        points = (
            discovered.select(group=PLUGIN_ENTRY_POINT_GROUP)
            if hasattr(discovered, "select")
            else discovered.get(PLUGIN_ENTRY_POINT_GROUP, ())
        )
    except Exception as exc:
        yield "entry-point discovery", lambda exc=exc: (_ for _ in ()).throw(exc)
        points = ()
    for point in sorted(points, key=lambda item: (item.name, item.value)):
        yield point.name, point.load
    for reference in filter(None, (
        item.strip() for item in os.environ.get(PLUGIN_MODULES_ENV, "").split(",")
    )):
        normalized = reference if ":" in reference else f"{reference}:plugin"
        yield reference, lambda normalized=normalized: load_object(normalized)
    try:
        records = _catalogue_installed()
    except Exception as exc:
        yield "catalogue installs", lambda exc=exc: (_ for _ in ()).throw(exc)
        records = {}
    for key, record in sorted(records.items()):
        if record.get("kind") == "plugin":
            yield key, lambda record=record: _load_catalogue_plugin(record)


def _build_registry() -> _Registry:
    """Discover valid contributions while recording each isolated load failure."""
    registry = _Registry()
    if os.environ.get(DISABLE_PLUGINS_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return registry
    app_keys: set[str] = set()
    model_keys: set[str] = set()
    report_keys: set[str] = set()
    loaded: List[SpacrPlugin] = []
    models: List[Tuple[str, ModelProviderContribution]] = []
    reports: List[Tuple[str, ReportSectionContribution]] = []
    for source, loader in _installed_sources():
        try:
            plugin = _coerce_plugin(loader())
            for app in plugin.apps:
                if app.key in app_keys:
                    raise ValueError(f"app key {app.key!r} is already registered")
                app_keys.add(app.key)
                registry.apps[app.key] = app
            for model in plugin.model_providers:
                if model.key in model_keys:
                    raise ValueError(f"model provider {model.key!r} is already registered")
                model_keys.add(model.key)
                models.append((plugin.name, model))
            for report in plugin.report_sections:
                if report.key in report_keys:
                    raise ValueError(f"report section {report.key!r} is already registered")
                report_keys.add(report.key)
                reports.append((plugin.name, report))
            loaded.append(plugin)
        except Exception as exc:
            diagnostic = PluginDiagnostic(
                source, "error", f"Could not load plugin {source!r}", repr(exc)
            )
            registry.diagnostics.append(diagnostic)
            LOG.exception("Could not load spaCR plugin %s", source)
    registry.plugins = tuple(loaded)
    registry.models = tuple(models)
    registry.reports = tuple(reports)
    return registry


def _registry() -> _Registry:
    """Return the lazily built process-wide plugin registry under its lock."""
    global _REGISTRY
    with _LOCK:
        if _REGISTRY is None:
            _REGISTRY = _build_registry()
        return _REGISTRY


def discover_plugins() -> Tuple[SpacrPlugin, ...]:
    """Return every valid discovered plugin in deterministic order.

    Returns an empty tuple, with no diagnostic recorded, when
    ``SPACR_DISABLE_PLUGINS`` is set to ``1``, ``true``, ``yes`` or ``on``
    (case-insensitively): nothing is imported at all. The result is cached;
    :func:`reload_plugins` discards the cache and discovers again.
    """
    return _registry().plugins


def reload_plugins() -> Tuple[SpacrPlugin, ...]:
    """Clear the discovery cache and discover again (primarily for tests/dev)."""
    global _REGISTRY
    with _LOCK:
        _REGISTRY = None
    return discover_plugins()


def plugin_apps() -> Tuple[AppContribution, ...]:
    """Return all contributed applications."""
    return tuple(_registry().apps.values())


def get_app(key: str) -> Optional[AppContribution]:
    """Return a contributed app by key, or ``None``.

    :param key: ``key`` of an installed plugin's :class:`AppContribution`;
        converted with ``str()`` before the registry lookup.
    """
    return _registry().apps.get(str(key))


def model_providers() -> Tuple[Tuple[str, ModelProviderContribution], ...]:
    """Return ``(plugin_name, provider)`` model-zoo contributions."""
    return _registry().models


def report_sections() -> Tuple[Tuple[str, ReportSectionContribution], ...]:
    """Return ``(plugin_name, section)`` report contributions."""
    return _registry().reports


def diagnostics() -> Tuple[PluginDiagnostic, ...]:
    """Return discovery and runtime contribution failures."""
    return tuple(_registry().diagnostics)


def record_diagnostic(
    plugin: str, message: str, exception: Any = "", severity: str = "error"
) -> None:
    """Record a model/report/runtime plugin failure without aborting spaCR.

    :param plugin: name of the plugin that failed, stored on the
        :class:`PluginDiagnostic` and written to the log.
    :param message: user-facing account of the failed operation.
    """
    diagnostic = PluginDiagnostic(
        str(plugin), str(severity), str(message), str(exception or "")
    )
    with _LOCK:
        _registry().diagnostics.append(diagnostic)
    LOG.error("%s: %s%s", plugin, message, f" ({exception})" if exception else "")


@dataclass(frozen=True)
class _CatalogueEntry:
    """One plugin or recipe offered by a catalogue.

    :param kind: ``"plugin"`` or ``"recipe"``.
    :param key: unique lower-case identifier used for the install folder.
    :param name: human-readable name shown in the browser.
    :param version: the version the catalogue offers.
    :param author: who wrote and maintains it.
    :param licence: its licence, as the author states it.
    :param summary: one or two sentences on what it does.
    :param homepage: where to read more, or empty.
    :param source: for a plugin, the pip requirement, wheel or folder to
        install; for a recipe, a settings JSON file when ``settings`` is empty.
    :param entry: for a plugin, the ``module:attribute`` of its manifest.
    :param api_version: the plugin SDK version a plugin targets.
    :param requirements: libraries a plugin needs beyond spaCR, installed
        into its own folder.
    :param app: for a recipe, the spaCR module its settings are for.
    :param settings: for a recipe, the settings it fills in.
    :param sha256: optional checksum of a local or downloaded source file.
    """

    kind: str
    key: str
    name: str
    version: str
    author: str = ""
    licence: str = ""
    summary: str = ""
    homepage: str = ""
    source: str = ""
    entry: str = ""
    api_version: str = PLUGIN_API_VERSION
    requirements: Tuple[str, ...] = ()
    app: str = ""
    settings: Mapping[str, Any] = field(default_factory=dict)
    sha256: str = ""


def _plugin_home(home: Any = None) -> str:
    """The folder catalogue installs live in.

    :param home: an explicit folder, which wins.
    :returns: ``$SPACR_PLUGIN_HOME`` when set, else ``~/.spacr/plugins``.
    """
    if home:
        return os.path.abspath(os.path.expanduser(str(home)))
    configured = os.environ.get(_PLUGIN_HOME_ENV, "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(os.path.expanduser("~"), ".spacr", "plugins")


def _is_url(source: str) -> bool:
    """Whether ``source`` is an http(s) address rather than a local path."""
    return str(source).lower().startswith(("http://", "https://"))


def _catalogue_location(source: Any = None) -> str:
    """Resolve a catalogue argument to a JSON file path or an address.

    :param source: a catalogue file, a folder holding ``catalogue.json``, an
        http(s) address, or None for ``$SPACR_PLUGIN_CATALOGUE``.
    :raises ValueError: when no catalogue is given or configured.
    """
    source = str(source or os.environ.get(_CATALOGUE_ENV, "")).strip()
    if not source:
        raise ValueError(
            f"no catalogue given; pass one or set {_CATALOGUE_ENV}")
    if _is_url(source):
        return source
    source = os.path.abspath(os.path.expanduser(source))
    if os.path.isdir(source):
        source = os.path.join(source, _CATALOGUE_FILE)
    return source


def _read_bytes(location: str) -> bytes:
    """Read a local file or download an http(s) address."""
    if _is_url(location):
        from urllib.request import urlopen

        with urlopen(location, timeout=60) as response:
            return response.read()
    with open(location, "rb") as handle:
        return handle.read()


def _resolve_source(base: str, source: str) -> str:
    """Resolve a source written relative to its catalogue.

    Pip requirements such as ``name>=1.0`` are returned unchanged; a relative
    path that exists next to the catalogue becomes absolute.
    """
    if not source or _is_url(source) or os.path.isabs(source):
        return source
    if _is_url(base):
        from urllib.parse import urljoin

        return urljoin(base, source)
    candidate = os.path.join(os.path.dirname(base), source)
    return os.path.abspath(candidate) if os.path.exists(candidate) else source


def _entry_from_mapping(value: Any, base: str) -> _CatalogueEntry:
    """Validate one catalogue row and return it with its sources resolved.

    :raises ValueError: for a missing field, an unknown kind or a bad key.
    """
    if not isinstance(value, Mapping):
        raise TypeError("each catalogue entry must be a mapping")
    data = {str(k): v for k, v in value.items()}
    unknown = set(data) - set(_CatalogueEntry.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown catalogue fields {sorted(unknown)}")
    for name in ("kind", "key", "name", "version"):
        if not str(data.get(name, "")).strip():
            raise ValueError(f"catalogue entry is missing {name!r}")
    if data["kind"] not in _CATALOGUE_KINDS:
        raise ValueError(f"catalogue kind must be one of {_CATALOGUE_KINDS}")
    if not _KEY_RE.match(str(data["key"])):
        raise ValueError(f"invalid catalogue key {data['key']!r}")
    data["requirements"] = _tuple_strings(
        data.get("requirements"), "requirements")
    settings = data.get("settings") or {}
    if not isinstance(settings, Mapping):
        raise TypeError("recipe settings must be a mapping")
    data["settings"] = dict(settings)
    for name in ("key", "name", "version", "author", "licence", "summary",
                 "homepage", "source", "entry", "api_version", "app",
                 "sha256"):
        if name in data:
            data[name] = str(data[name]).strip()
    entry = _CatalogueEntry(**data)
    if entry.kind == "plugin":
        if not _REF_RE.match(entry.entry):
            raise ValueError(
                f"plugin {entry.key!r} needs entry 'package.module:attribute'")
        if not entry.source:
            raise ValueError(f"plugin {entry.key!r} has no source to install")
    elif not entry.settings and not entry.source:
        raise ValueError(f"recipe {entry.key!r} has neither settings nor source")
    return _CatalogueEntry(**{
        **entry.__dict__, "source": _resolve_source(base, entry.source)})


def _read_catalogue(source: Any = None) -> Tuple[_CatalogueEntry, ...]:
    """Read and validate a catalogue of plugins and recipes.

    The catalogue is a JSON object with a ``"plugins"`` and a ``"recipes"``
    list; each row gives at least a key, name and version, and the author,
    licence and summary the browser shows.

    :param source: see :func:`_catalogue_location`.
    :returns: the entries, plugins first, each sorted by name.
    :raises ValueError: for a malformed catalogue or a repeated key.
    """
    import json

    location = _catalogue_location(source)
    data = json.loads(_read_bytes(location).decode("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("a catalogue must be a JSON object")
    entries: List[_CatalogueEntry] = []
    for kind, rows in (("plugin", data.get("plugins", ())),
                       ("recipe", data.get("recipes", ()))):
        if not isinstance(rows, Sequence) or isinstance(rows, str):
            raise ValueError(f"catalogue {kind}s must be a list")
        entries.extend(sorted(
            (_entry_from_mapping({"kind": kind, **row}, location)
             for row in rows),
            key=lambda item: item.name.lower()))
    keys = [entry.key for entry in entries]
    repeated = next((key for key in keys if keys.count(key) > 1), "")
    if repeated:
        raise ValueError(f"catalogue repeats key {repeated!r}")
    return tuple(entries)


def _catalogue_installed(home: Any = None) -> Dict[str, Dict[str, Any]]:
    """The catalogue installs recorded in the plugin home, by key."""
    import json

    path = os.path.join(_plugin_home(home), _INSTALLED_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): dict(value) for key, value in dict(data).items()}


def _write_installed(records: Mapping[str, Any], home: Any = None) -> None:
    """Replace the install record atomically."""
    import json

    root = _plugin_home(home)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, _INSTALLED_FILE)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(records), handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _newer(offered: str, installed: str) -> bool:
    """Whether version ``offered`` is later than ``installed``."""
    try:
        from packaging.version import Version

        return Version(offered) > Version(installed)
    except Exception:
        return offered != installed


def _catalogue_rows(source: Any = None, home: Any = None) -> List[Dict[str, str]]:
    """What the catalogue browser shows: every entry and its install state.

    :param source: see :func:`_catalogue_location`.
    :param home: the plugin home, see :func:`_plugin_home`.
    :returns: one dict per entry with its metadata, the installed version
        and a ``status`` of ``"available"``, ``"installed"``,
        ``"update available"`` or ``"incompatible"``.
    """
    installed = _catalogue_installed(home)
    rows = []
    for entry in _read_catalogue(source):
        record = installed.get(entry.key, {})
        have = str(record.get("version", ""))
        if entry.kind == "plugin" and entry.api_version.split(".", 1)[0] != (
                PLUGIN_API_VERSION.split(".", 1)[0]):
            status = "incompatible"
        elif not have:
            status = "available"
        elif _newer(entry.version, have):
            status = "update available"
        else:
            status = "installed"
        rows.append({
            "kind": entry.kind, "key": entry.key, "name": entry.name,
            "version": entry.version, "installed": have,
            "author": entry.author, "licence": entry.licence,
            "summary": entry.summary, "homepage": entry.homepage,
            "app": entry.app, "status": status,
        })
    return rows


def _checked_bytes(entry: _CatalogueEntry, location: str) -> bytes:
    """Read ``location`` and check it against the entry's sha256, if given."""
    import hashlib

    payload = _read_bytes(location)
    if entry.sha256 and hashlib.sha256(payload).hexdigest() != entry.sha256.lower():
        raise ValueError(f"{entry.key!r}: {location} does not match its sha256")
    return payload


def _forget_modules_under(folder: str) -> None:
    """Drop imported modules loaded from ``folder`` and its path entry."""
    import sys

    folder = os.path.abspath(folder)
    for name, module in list(sys.modules.items()):
        where = getattr(module, "__file__", None) or ""
        if where and os.path.abspath(where).startswith(folder + os.sep):
            sys.modules.pop(name, None)
    while folder in sys.path:
        sys.path.remove(folder)
    importlib.invalidate_caches()


def _load_catalogue_plugin(record: Mapping[str, Any]) -> Any:
    """Load an installed catalogue plugin from its private folder.

    The folder is appended to ``sys.path``, so a package spaCR already has
    always wins over a copy in the plugin's folder.
    """
    import sys

    folder = str(record["path"])
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"plugin folder {folder} is missing")
    if folder not in sys.path:
        sys.path.append(folder)
        importlib.invalidate_caches()
    return load_object(str(record["entry"]))


def _pip(arguments: Sequence[str], runner: Optional[Callable[..., Any]] = None) -> None:
    """Run pip in spaCR's interpreter and raise with its output on failure."""
    import subprocess
    import sys

    command = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check", "--no-input", *arguments]
    result = (runner or subprocess.run)(
        command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "pip could not install the plugin:\n"
            + (result.stderr or result.stdout or "").strip()[-2000:])


def _install_plugin(entry: _CatalogueEntry, root: str,
                    runner: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Install one plugin into a fresh folder and check that it loads."""
    import shutil
    import sys

    if entry.api_version.split(".", 1)[0] != PLUGIN_API_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"{entry.name} needs plugin SDK {entry.api_version}; "
            f"spaCR provides {PLUGIN_API_VERSION}")
    site = os.path.join(root, "site")
    staging = os.path.join(site, f".{entry.key}.new")
    final = os.path.join(site, entry.key)
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    local = os.path.exists(entry.source)
    if local and os.path.isfile(entry.source):
        _checked_bytes(entry, entry.source)
    try:
        index = (["--no-index", "--find-links",
                  os.path.dirname(os.path.abspath(entry.source))]
                 if local else [])
        _pip(["--no-deps", "--target", staging, *index, entry.source], runner)
        if entry.requirements:
            _pip(["--target", staging, *entry.requirements], runner)
        sys.path.append(staging)
        importlib.invalidate_caches()
        try:
            plugin = _coerce_plugin(load_object(entry.entry))
        finally:
            _forget_modules_under(staging)
        _forget_modules_under(final)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"kind": "plugin", "version": entry.version, "entry": entry.entry,
            "path": final, "plugin": plugin.name, "name": entry.name}


def _install_recipe(entry: _CatalogueEntry, root: str) -> Dict[str, Any]:
    """Write one recipe's settings where a settings loader can read them."""
    import json

    settings = dict(entry.settings)
    if not settings:
        settings = json.loads(_checked_bytes(entry, entry.source).decode("utf-8"))
        if not isinstance(settings, dict):
            raise ValueError(f"recipe {entry.key!r} is not a settings object")
    folder = os.path.join(root, "recipes")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{entry.key}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True, default=str)
    return {"kind": "recipe", "version": entry.version, "app": entry.app,
            "path": path, "name": entry.name}


def _install_from_catalogue(key: str, source: Any = None, home: Any = None,
                            runner: Optional[Callable[..., Any]] = None
                            ) -> Dict[str, Any]:
    """Install or update one catalogue entry and reload the plugins.

    A plugin is installed into a staging folder, loaded and validated there,
    and only then swapped in for any earlier version, so a failed install or
    update leaves the previous one working. A recipe's settings are written
    to ``recipes/<key>.json`` in the plugin home, a file every settings
    loader reads.

    :param key: the entry's key in the catalogue.
    :param source: see :func:`_catalogue_location`.
    :param home: the plugin home, see :func:`_plugin_home`.
    :param runner: replaces ``subprocess.run`` for pip.
    :returns: the install record.
    :raises KeyError: when the catalogue has no such key.
    """
    entry = next((item for item in _read_catalogue(source) if item.key == key),
                 None)
    if entry is None:
        raise KeyError(f"the catalogue has no entry {key!r}")
    root = _plugin_home(home)
    with _LOCK:
        record = (_install_plugin(entry, root, runner) if entry.kind == "plugin"
                  else _install_recipe(entry, root))
        records = _catalogue_installed(root)
        records[entry.key] = {**record, "author": entry.author,
                              "licence": entry.licence}
        _write_installed(records, root)
    if entry.kind == "plugin":
        reload_plugins()
    return records[entry.key]


def _uninstall_from_catalogue(key: str, home: Any = None) -> bool:
    """Remove an installed catalogue plugin or recipe and reload the plugins.

    :param key: the entry's key.
    :param home: the plugin home, see :func:`_plugin_home`.
    :returns: False when nothing by that key was installed.
    """
    import shutil

    root = _plugin_home(home)
    with _LOCK:
        records = _catalogue_installed(root)
        record = records.pop(str(key), None)
        if record is None:
            return False
        path = str(record.get("path", ""))
        if record.get("kind") == "plugin":
            _forget_modules_under(path)
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
        _write_installed(records, root)
    if record.get("kind") == "plugin":
        reload_plugins()
    return True
