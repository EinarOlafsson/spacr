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


def test_swedish_reviewed_runtime_text_is_source_bound_and_gate_clean() -> None:
    from build_i18n_catalogs import (
        _looks_translatable,
        _translation_rejection_reasons,
        canonical_sources,
        reviewed_runtime_translations,
    )

    reviewed = reviewed_runtime_translations("sv")
    sources = canonical_sources()
    current_values = set(sources["setting_labels"].values())
    current_values.update(sources["setting_tooltips"].values())
    current_values.update(sources["ui"])

    # THE NUMBER FOLLOWS THE EVIDENCE, not the other way round. These count
    # the records under docs/i18n/reviewed/runtime/<lang>, and they last moved
    # on 2026-09-08 for the release: Swedish 116 -> 118, the two SETTING_LABELS
    # rows the build's own QA gate reported as still exact English
    # (`arr_axes`, `cellpose_diameter`). French is unchanged at 96. The loop
    # below is the actual contract: every record must still bind to a live
    # source value and still pass the current syntax, semantic, script and
    # exact-copy gates, so a record that has drifted fails here rather than
    # being absorbed by a looser count.
    assert len(reviewed) == 118
    for source, translated in reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "sv",
            force=_looks_translatable(source),
        )


def test_french_reviewed_runtime_text_is_source_bound_and_gate_clean() -> None:
    from build_i18n_catalogs import (
        _looks_translatable,
        _translation_rejection_reasons,
        canonical_sources,
        reviewed_runtime_translations,
    )

    reviewed = reviewed_runtime_translations("fr")
    sources = canonical_sources()
    current_values = set(sources["setting_labels"].values())
    current_values.update(sources["setting_tooltips"].values())
    current_values.update(sources["ui"])

    # THE NUMBER FOLLOWS THE EVIDENCE, not the other way round. These count
    # the records under docs/i18n/reviewed/runtime/<lang>, and they last moved
    # on 2026-09-08 for the release: French 96 -> 97, the single row
    # (`meta_regex`) the build's own QA gate still reported as exact English.
    #
    # ITS FIRST DRAFT WAS REJECTED, and for the right reason: "si bien que"
    # put the ADVERB sense of "well" into a caption whose subject is the plate
    # well, which is precisely what `scientific-well-as-adverb` exists to
    # catch. Rephrased to "de sorte que".
    #
    # 97 -> 99 later the same day, +5/-3. The five are the cellprob-threshold
    # tooltips the work session's ranker found reading "gouttes" -- droplets
    # -- where the English says the threshold DROPS faint organelles. They
    # replace three records rather than adding five, because all four
    # `organelle*_cellprob_threshold` tooltips share a byte-identical English
    # source across the slots and the store keys by SOURCE: four slot-numbered
    # translations of one string is a contradiction, and it is rejected as
    # one. They now share a single target.
    #
    # The loop below is the actual contract: every record must still bind to a
    # live source value and still pass the current syntax, semantic, script
    # and exact-copy gates, so a record that has drifted fails here rather
    # than being absorbed by a looser count.
    # 99 -> 98 on 2026-09-11, +1/-2, and the number follows the evidence as
    # this note says it must.
    #
    #   GONE, both of them tooltips for settings 364 deleted: the legacy
    #   Image-UMAP crop selector ("the current workflow always joins
    #   measurement tables...") and one of the legacy compatibility fields
    #   ("current plotting paths do not read it"). A reviewed record for a
    #   source that is no longer in the catalog is not evidence about
    #   anything, and it left with the setting.
    #
    #   ADDED: 'press Escape to close'. The rebuilt catalogs had replaced a
    #   good row with "Appuyez sur Échapper pour fermer" -- the key name
    #   translated as the verb -- in eight of the nine languages, and
    #   nothing claimed the row, so the rebuild was free to. It is claimed
    #   now.
    assert len(reviewed) == 98
    for source, translated in reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "fr",
            force=_looks_translatable(source),
        )
