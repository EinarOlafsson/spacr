"""Localized one-line descriptions for spaCR's built-in modules.

The stable application key is the lookup key.  Keeping these longer strings
outside the compact UI catalog makes them reviewable by fluent speakers and
prevents term-by-term translation from corrupting scientific descriptions.
"""

from __future__ import annotations

from typing import Optional

from .i18n_module_summaries_asia import MODULE_SUMMARIES_ASIA
from .i18n_module_summaries_other import MODULE_SUMMARIES_OTHER
from .i18n_module_summaries_west import MODULE_SUMMARIES_WEST


MODULE_SUMMARIES = {
    **MODULE_SUMMARIES_WEST,
    **MODULE_SUMMARIES_ASIA,
    **MODULE_SUMMARIES_OTHER,
}


def module_summary(
    app_key: str,
    english: str,
    language: Optional[str] = None,
) -> str:
    """Return a reviewed module summary, falling back to ``english``.

    Plugin modules and future built-ins therefore remain readable until their
    own translation catalog supplies an exact description.
    """
    from .i18n import _exact_translation, current_language, normalize_language

    code = normalize_language(language or current_language())
    if code == "en":
        return str(english)
    reviewed = MODULE_SUMMARIES.get(code, {}).get(str(app_key))
    if reviewed:
        return reviewed
    # Plugins may ship exact translations in their manifest.  Do not apply
    # conservative term substitution to a scientific paragraph: either the
    # plugin supplies the whole sentence or it stays canonical English.
    return _exact_translation(str(english), code) or str(english)


def validate_module_summaries() -> None:
    """Raise if the nine non-English catalogs drift out of alignment."""
    expected_languages = {
        "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
    }
    assert set(MODULE_SUMMARIES) == expected_languages
    key_sets = {frozenset(items) for items in MODULE_SUMMARIES.values()}
    assert len(key_sets) == 1
    assert len(next(iter(key_sets))) == 34


validate_module_summaries()


__all__ = ["MODULE_SUMMARIES", "module_summary"]
