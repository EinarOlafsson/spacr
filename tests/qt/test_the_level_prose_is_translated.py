"""286: the level tooltips and hardware notes read in every language.

"Runtime/API translation ratchets include the five names, tooltips and
migration messages in all nine supported languages."

The five names were already translated. The five tooltips and the five
hardware notes were not, in ANY language: the runtime extractor iterated
the two dicts and so collected their keys ("laptop", ...) instead of the
prose. The extractor now collects the values, and each string has a
hand-written reviewed record in all nine languages.

THERE ARE NO MIGRATION MESSAGES TO TRANSLATE. The migration is silent by
design -- it keeps the user's choice, so it has nothing to tell them -- and
it logs at DEBUG only. That half of the bullet is recorded in the item as a
question for the maintainer, not closed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import memory_budget as mb
from spacr.qt import preferences as P

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
PROSE = (*P.PERFORMANCE_NOTES.values(), *mb.HARDWARE_NOTES.values())


def test_the_level_prose_reaches_the_runtime_catalog_sources():
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import build_i18n_catalogs as builder

    sources = builder._indirect_runtime_ui_sources()
    for text in PROSE:
        assert text in sources, f"never extracted for translation: {text!r}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_level_string_is_translated(language):
    from spacr.qt.i18n import _exact_translation

    missing = [text for text in (*P.PERFORMANCE_LABELS.values(), *PROSE)
               if not _exact_translation(text, language)]
    assert missing == [], (
        f"{language} shows these performance-level strings in English:\n"
        + "\n".join(missing))
