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
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    """One runnable GUI/headless application contributed by a plugin."""

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
    """A callable returning model-zoo entries or entry mappings."""

    key: str
    provider: str


@dataclass(frozen=True)
class ReportSectionContribution:
    """A callable adding one section to :func:`spacr.report.collect_report`."""

    key: str
    title: str
    builder: str
    after: str = "statistics"


@dataclass(frozen=True)
class SpacrPlugin:
    """Validated plugin manifest returned by a ``spacr.plugins`` entry point."""

    name: str
    version: str
    api_version: str = PLUGIN_API_VERSION
    apps: Tuple[AppContribution, ...] = ()
    model_providers: Tuple[ModelProviderContribution, ...] = ()
    report_sections: Tuple[ReportSectionContribution, ...] = ()
    translations: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginDiagnostic:
    """One discovery or contribution error visible to users and logs."""

    plugin: str
    severity: str
    message: str
    exception: str = ""


@dataclass(frozen=True)
class ReportContext:
    """Read-only inputs passed to plugin report-section builders."""

    src: Any
    artifacts: Mapping[str, Any]
    runs: Tuple[Mapping[str, Any], ...]
    options: Mapping[str, Any]


@dataclass
class _Registry:
    plugins: Tuple[SpacrPlugin, ...] = ()
    apps: Dict[str, AppContribution] = field(default_factory=dict)
    models: Tuple[Tuple[str, ModelProviderContribution], ...] = ()
    reports: Tuple[Tuple[str, ReportSectionContribution], ...] = ()
    diagnostics: List[PluginDiagnostic] = field(default_factory=list)


_LOCK = threading.RLock()
_REGISTRY: Optional[_Registry] = None


def load_object(reference: str) -> Any:
    """Import and return ``module:attribute`` (nested attributes supported)."""
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
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} cannot contain blank values")
    return result


def _mapping_of_strings(value: Any, field_name: str) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _app_from_mapping(value: Any) -> AppContribution:
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
    """Validate a mapping and return its immutable :class:`SpacrPlugin`."""
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
    if callable(value) and not isinstance(value, type):
        value = value()
    if isinstance(value, SpacrPlugin):
        _validate_plugin(value)
        return value
    if isinstance(value, Mapping):
        return plugin_from_mapping(value)
    raise TypeError("entry point must expose SpacrPlugin, a manifest mapping, or a factory")


def _installed_sources() -> Iterable[Tuple[str, Callable[[], Any]]]:
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


def _build_registry() -> _Registry:
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
    global _REGISTRY
    with _LOCK:
        if _REGISTRY is None:
            _REGISTRY = _build_registry()
        return _REGISTRY


def discover_plugins() -> Tuple[SpacrPlugin, ...]:
    """Return every valid discovered plugin in deterministic order."""
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
    """Return a contributed app by key, or ``None``."""
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
    """Record a model/report/runtime plugin failure without aborting spaCR."""
    diagnostic = PluginDiagnostic(
        str(plugin), str(severity), str(message), str(exception or "")
    )
    with _LOCK:
        _registry().diagnostics.append(diagnostic)
    LOG.error("%s: %s%s", plugin, message, f" ({exception})" if exception else "")
