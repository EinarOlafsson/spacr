"""Translated presentations of the End User Licence Agreement.

The agreement is not in the caption catalogs: it is 7,621 characters.
Nine locales of it would be roughly double the entire existing runtime
catalog, and it would flow into every ratchet, count and coverage figure the
translation tooling tracks -- so those numbers would stop describing the user
interface and start being mostly licence text.

A presentation registered here is a convenience presentation of the English
agreement. Section 11.4 of the agreement says so in its own words: if
a translation differs from the English in any respect, the English governs.
Nothing here is a separate agreement and nothing here is what a profile
accepts -- `terms.record_agreement` stores `TERMS_VERSION`, which is the
English document's version, whichever language was on screen.

Review status is recorded per locale, through the ``review`` argument of
:func:`register`, rather than asserted collectively, and :func:`register`
refuses an empty one. A locale nobody read must not be recorded as reviewed.
No locale presentation is registered in this package at present, so every
language shows the English agreement.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: ``language code -> (paragraphs, review note)``. Filled by the modules
#: imported below; a locale with no module simply has no entry and falls
#: back to English, which is always correct here by Section 11.4.
_TRANSLATIONS: Dict[str, Tuple[Tuple[str, ...], str]] = {}


def register(code: str, paragraphs: Tuple[str, ...], review: str) -> None:
    """Record one locale's presentation of the agreement.

    :param code: language code, as :mod:`spacr.qt.i18n` spells it.
    :param paragraphs: the agreement's clauses, in the English order.
    :param review: who reviewed it and when, or an explicit statement that
        it is a machine draft. Never left empty.
    """
    if not review:
        raise ValueError(
            f"{code}: a translated agreement must record its review status")
    _TRANSLATIONS[str(code)] = (tuple(paragraphs), str(review))


def available() -> Tuple[str, ...]:
    """Language codes that have a translated presentation."""
    return tuple(sorted(_TRANSLATIONS))


def paragraphs(code: str):
    """The agreement in ``code``, or ``None`` to use the English.

    :param code: language code, as :mod:`spacr.qt.i18n` spells it; a code
        with no registered translation gives ``None``.
    """
    entry = _TRANSLATIONS.get(str(code))
    return entry[0] if entry else None


def review_note(code: str) -> str:
    """Who reviewed ``code``'s presentation, or ``""`` if it has none.

    :param code: language code, as :mod:`spacr.qt.i18n` spells it.
    """
    entry = _TRANSLATIONS.get(str(code))
    return entry[1] if entry else ""
