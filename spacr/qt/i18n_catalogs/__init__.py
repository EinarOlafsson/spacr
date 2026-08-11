"""External localization catalogs for long and high-volume UI text.

The compact, hand-reviewed chrome catalog remains in :mod:`spacr.qt.i18n`.
Setting labels, scientific tooltip bodies, category explanations and the
remaining static Qt text are much larger, so they live in one generated,
reviewable module per language in this package.  Keeping them outside widget
code also means a translation correction never touches analytical logic.

Every setting translation is checked against the canonical English source in
``en.py`` before it is returned.  A renamed setting or edited tooltip therefore
falls back to current English instead of silently showing an obsolete
translation.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Optional


CATALOG_LANGUAGES = (
    "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
)


@lru_cache(maxsize=None)
def _module(language: str) -> Optional[ModuleType]:
    """Return one language module, or ``None`` for English/unknown codes."""
    code = str(language or "")
    if code not in CATALOG_LANGUAGES:
        return None
    name = f"{__name__}.{code}"
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        # A missing optional catalog is a safe English fallback.  Do not hide
        # an import failure *inside* a catalog module, which is a real defect.
        if exc.name == name:
            return None
        raise


@lru_cache(maxsize=1)
def _english() -> ModuleType:
    return import_module(f"{__name__}.en")


def ui_text(source: str, language: str) -> Optional[str]:
    """Return an exact translation for spaCR-owned static Qt text."""
    module = _module(language)
    if module is None:
        return None
    value = getattr(module, "UI", {}).get(str(source))
    return str(value) if isinstance(value, str) and value.strip() else None


def setting_label(
    key: str,
    source: str,
    language: str,
    app_key: str = "",
) -> Optional[str]:
    """Return a label translation only while its English source still matches."""
    canonical = getattr(_english(), "SETTING_LABELS", {})
    lookup = f"{app_key}.{key}" if app_key else str(key)
    if canonical.get(lookup) != str(source):
        lookup = str(key)
    if canonical.get(lookup) != str(source):
        return None
    module = _module(language)
    if module is None:
        return None
    value = getattr(module, "SETTING_LABELS", {}).get(lookup)
    return str(value) if isinstance(value, str) and value.strip() else None


def setting_tooltip(
    key: str,
    source: str,
    language: str,
) -> Optional[str]:
    """Return a scientific tooltip only while the canonical prose matches."""
    canonical = getattr(_english(), "SETTING_TOOLTIPS", {})
    if canonical.get(str(key)) != str(source):
        return None
    module = _module(language)
    if module is None:
        return None
    value = getattr(module, "SETTING_TOOLTIPS", {}).get(str(key))
    return str(value) if isinstance(value, str) and value.strip() else None


def category_help(source: str, language: str) -> Optional[str]:
    """Return an exact translation for a written settings-category blurb."""
    text = str(source)
    if text not in getattr(_english(), "CATEGORY_SOURCES", frozenset()):
        return None
    module = _module(language)
    if module is None:
        return None
    value = getattr(module, "CATEGORY_HELP", {}).get(text)
    return str(value) if isinstance(value, str) and value.strip() else None


def module_summary(
    key: str,
    source: str,
    language: str,
) -> Optional[str]:
    """Return a module summary while its canonical description still matches."""
    canonical = getattr(_english(), "MODULE_SUMMARIES", {})
    if canonical.get(str(key)) != str(source):
        return None
    module = _module(language)
    if module is None:
        return None
    value = getattr(module, "MODULE_SUMMARIES", {}).get(str(key))
    return str(value) if isinstance(value, str) and value.strip() else None


__all__ = [
    "CATALOG_LANGUAGES",
    "category_help",
    "setting_label",
    "setting_tooltip",
    "module_summary",
    "ui_text",
]
