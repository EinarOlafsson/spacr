"""Focused syntax contracts for generated runtime localization catalogs."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_swedish_example_abbreviation_is_not_a_dotted_identifier() -> None:
    from build_i18n_catalogs import _syntax_preserved

    source = "Choose a column, e.g. 'plateID', from measurements.db."
    translated = "Välj en kolumn, t.ex. 'plateID', från measurements.db."

    assert _syntax_preserved(source, translated)
    assert not _syntax_preserved(
        source,
        translated.replace("measurements.db", "measurements.database"),
    )


def test_swedish_reviewed_tooltips_are_source_bound_and_gate_clean() -> None:
    from build_i18n_catalogs import (
        _looks_translatable,
        _translation_rejection_reasons,
        canonical_sources,
        reviewed_runtime_translations,
    )

    reviewed = reviewed_runtime_translations("sv")
    sources = canonical_sources()["setting_tooltips"]

    assert len(reviewed) == 19
    for source, translated in reviewed.items():
        assert source in sources.values()
        assert not _translation_rejection_reasons(
            source,
            translated,
            "sv",
            force=_looks_translatable(source),
        )
