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

import hashlib
import re
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


def _localized_value(
    module: ModuleType,
    table_name: str,
    key: str,
    source: str,
) -> Optional[str]:
    """Return a localized record only when its stored source hash is current."""
    expected = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    hashes = getattr(module, "SOURCE_HASHES", {})
    if hashes.get((table_name, str(key))) != expected:
        return None
    value = getattr(module, table_name, {}).get(str(key))
    return str(value) if isinstance(value, str) and value.strip() else None


#: A bare snake_case name -- ``image_path``, ``fdr_bh``, ``RdBu_r``. These
#: reach the catalogs because they are shown to the user, in a combo box or
#: a status line, but they are NAMES rather than prose: a column, a
#: correction method, a colour map. Whatever a translation model does to one
#: it stops naming the thing it named, and where the caption is read back to
#: choose the column it also stops matching. Measured in the shipped
#: catalogs: ``image_path`` was stored as ``image_path.`` in four languages,
#: ``png_list E-mail`` in Portuguese and ``RdBu_r( 빈 공간)`` in Korean.
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")


def ui_text(source: str, language: str) -> Optional[str]:
    """Return an exact translation for spaCR-owned static Qt text.

    An identifier answers with itself: see :data:`_IDENTIFIER`.
    """
    text = str(source)
    if _IDENTIFIER.match(text):
        return None
    module = _module(language)
    if module is None:
        return None
    return _localized_value(module, "UI", text, text)


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
        # Slots above the four default organelles reuse the primary slot's
        # reviewed translation.  Materialising all 26 otherwise copies the
        # same 53 labels and tooltips 22 extra times into every language
        # catalog.  Accept the alias only when both its key and its exact
        # generated English label match; edited prose must still fall back to
        # English instead of displaying a stale translation.
        default_limit = 4
        try:
            from spacr.organelle_types import (
                CATALOGUED_ORGANELLE_SLOTS,
                organelle_number,
                organelle_role_of,
                primary_setting,
            )

            role = organelle_role_of(str(key))
            number = organelle_number(role) if role else 0
            default_limit = CATALOGUED_ORGANELLE_SLOTS
            primary = primary_setting(str(key))
            primary_source = canonical.get(primary)
            expected = (
                str(primary_source).replace(
                    "Organelle 1", f"Organelle {number}", 1
                )
                if primary_source is not None else None
            )
        except (ImportError, TypeError, ValueError):
            role = None
            number = 0
            primary = ""
            primary_source = None
            expected = None
        if not (
            role
            and number > default_limit
            and expected == str(source)
        ):
            return None
        module = _module(language)
        if module is None:
            return None
        localized = _localized_value(
            module, "SETTING_LABELS", primary, str(primary_source)
        )
        if localized is None:
            return None
        return re.sub(r"(?<!\d)1(?!\d)", str(number), localized, count=1)
    module = _module(language)
    if module is None:
        return None
    return _localized_value(
        module, "SETTING_LABELS", lookup, canonical[lookup]
    )


def setting_tooltip(
    key: str,
    source: str,
    language: str,
    app_key: str = "",
) -> Optional[str]:
    """Return a scientific tooltip only while the canonical prose matches.

    App-specific records take precedence when the same setting key has a
    different meaning in one module. A shared record remains the fallback for
    callers that provide an app key but use the canonical global description.
    """
    canonical = getattr(_english(), "SETTING_TOOLTIPS", {})
    lookup = f"{app_key}.{key}" if app_key else str(key)
    if canonical.get(lookup) != str(source):
        lookup = str(key)
    if canonical.get(lookup) != str(source):
        default_limit = 4
        try:
            from spacr.organelle_types import (
                CATALOGUED_ORGANELLE_SLOTS,
                organelle_number,
                organelle_role_of,
                primary_setting,
            )

            role = organelle_role_of(str(key))
            number = organelle_number(role) if role else 0
            default_limit = CATALOGUED_ORGANELLE_SLOTS
            primary = primary_setting(str(key))
            primary_source = canonical.get(primary)
            expected = str(primary_source)
            expected = expected.replace("organelle_", f"{role}_")
            expected = expected.replace(
                "organelle ", f"organelle {number} "
            )
        except (ImportError, TypeError, ValueError):
            role = None
            number = 0
            primary = ""
            primary_source = None
            expected = None
        if not (
            role
            and number > default_limit
            and expected == str(source)
        ):
            return None
        module = _module(language)
        if module is None:
            return None
        localized = _localized_value(
            module, "SETTING_TOOLTIPS", primary, str(primary_source)
        )
        if localized is None:
            return None
        return localized.replace("organelle_", f"{role}_")
    module = _module(language)
    if module is None:
        return None
    return _localized_value(
        module, "SETTING_TOOLTIPS", lookup, canonical[lookup]
    )


def category_help(source: str, language: str) -> Optional[str]:
    """Return an exact translation for a written settings-category blurb."""
    text = str(source)
    if text not in getattr(_english(), "CATEGORY_SOURCES", frozenset()):
        return None
    module = _module(language)
    if module is None:
        return None
    return _localized_value(module, "CATEGORY_HELP", text, text)


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
    return _localized_value(
        module, "MODULE_SUMMARIES", str(key), canonical[str(key)]
    )


__all__ = [
    "CATALOG_LANGUAGES",
    "category_help",
    "setting_label",
    "setting_tooltip",
    "module_summary",
    "ui_text",
]
