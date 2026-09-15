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
    # 118 -> 134 on 2026-09-15, +16/-0: the live magnifier's captions (407),
    # written by hand as reviewed records so its first catalog build needed
    # no model. 'Cellpose' has no record: a name is kept as it is by the
    # gates, and an identity record is rejected as an exact copy.
    #
    # 134 -> 191 on 2026-09-15, +57/-0, for the 08:15 integration batch
    # (wip/integ-0815) on nightly 554dcf371:
    #    2  the `segmentation_backend` label and tooltip (404/405),
    #       2026-09-15-segmentation-backend.json
    #   55  corrections from reading every row the batch's GPU pass changed,
    #       2026-09-15-integration-review.json -- including the Cellpose 4
    #       diameter-check UI row the model left English, whose reviewed
    #       translation replaced the hand-written one first recorded in
    #       2026-09-15-integration-new.json (that file is gone)
    # PROVED BY SUBTRACTION with the loader itself: the live count minus the
    # sources in those two files is 134, they share no source with each other,
    # and neither shares one with any other file.
    #
    # 191 -> 201 on 2026-09-15, +10/-0, for item 286: the five performance
    # level tooltips and the five hardware notes, hand-written in
    # 2026-09-15-performance-levels.json. None had ever reached a translator:
    # the runtime extractor iterated the two dicts and collected their keys.
    # All ten sources are new wording, so the file shares no source with any
    # other, and the live count minus its ten is 191.
    #
    # 201 -> 219 on 2026-09-15, +21/-3, for the magnifier's border option and
    # whole-image mode (407), rebased onto nightly df1216b3f. The 21 are
    # 2026-09-15-magnifier-whole-image.json; the 3 left
    # 2026-09-15-live-magnifier.json with the card subtitle, the Size tooltip
    # and the Magnifier-button tooltip, whose English was reworded because the
    # whole-image mode made it untrue. PROVED BY SUBTRACTION with the loader:
    # 201 + 21 - 3 = 219 is the live count, the live count minus that file's
    # sources is 198 (201 - 3), and the file shares no source with any other.
    # +2 on 2026-09-15, both from 387's selectivity index on the
    # Dose-Response screen: "Host response" and its tooltip, in
    # 2026-09-15-dose-response-host-readout.json. Staged for the catalog
    # pass rather than regenerated here, as the catalog lane asked.
    # +4 more on 2026-09-15, from 387's checkerboard scoring: "Second
    # compound", its tooltip, "Bliss independence" and "Loewe
    # additivity", in 2026-09-15-dose-response-combination.json.
    assert len(reviewed) == 225
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
    #
    # 98 -> 99 on 2026-09-15, +1. ADDED: the `resume` setting label,
    # 'Reprendre'. Five locales rendered "Resume" as the CV noun (resume ->
    # CV) rather than the verb the Run button means, and a rebuild would put
    # the noun back from the translation cache, so the fix is claimed as a
    # record rather than hand-edited into the catalog (items 397, 406).
    #
    # 99 -> 116 on 2026-09-15, +17/-0: the magnifier's sixteen captions
    # (407), plus 'Overlap', which read 'Rupture' (a break) on the same card
    # and is now 'Chevauchement'.
    #
    # 116 -> 169 on 2026-09-15, +53/-0, for the 08:15 integration batch:
    #    2  the `segmentation_backend` label and tooltip (404/405),
    #       2026-09-15-segmentation-backend.json
    #   51  corrections from reading every row the batch's GPU pass changed,
    #       2026-09-15-integration-review.json
    # The live count minus the sources in those two files is 116, and they
    # share a source with no other file or with each other.
    #
    # 169 -> 179 on 2026-09-15, +10/-0, for item 286: the same ten
    # performance-level strings as the Swedish note above, in
    # 2026-09-15-performance-levels.json. The live count minus its ten is 169.
    #
    # 179 -> 197 on 2026-09-15, +21/-3, for the magnifier's border option and
    # whole-image mode (407): the same 21 new records and the same 3 retired
    # wordings as Swedish. 179 + 21 - 3 = 197 is the live count, the live
    # count minus the new file's sources is 176 (179 - 3), and the file shares
    # no source with any other.
    # +2 on 2026-09-15, both from 387's selectivity index on the
    # Dose-Response screen: "Host response" and its tooltip, in
    # 2026-09-15-dose-response-host-readout.json. Staged for the catalog
    # pass rather than regenerated here, as the catalog lane asked.
    # +4 more on 2026-09-15, from 387's checkerboard scoring: "Second
    # compound", its tooltip, "Bliss independence" and "Loewe
    # additivity", in 2026-09-15-dose-response-combination.json.
    assert len(reviewed) == 203
    for source, translated in reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "fr",
            force=_looks_translatable(source),
        )
