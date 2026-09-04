# Localization review scope at the 1.5.0.5 release gate — 2026-09-04

This supersedes `REVIEW_SCOPE_2026-08-30.md`, which is kept as history. Like
that report, **this is an evidence report and not a certificate that every
sentence was read by a fluent speaker.** The catalogs have complete mechanical
source coverage; the semantic-review evidence remains defect-driven and much
smaller than the shipped corpus.

## What changed since 2026-08-30

The runtime inventory grew from 4,982 entries to **5,165**: 1,019 setting
labels, 1,014 setting tooltips, 192 category explanations, 2,817 UI strings, 57
installer strings and 66 module summaries. The API inventory grew from 8,861
symbol documents to **9,434**, all ten locales in agreement.

Every mechanical gate now passes. `tests/qt/test_external_i18n_catalogs.py` is
31 of 31, having been 7 failed / 24 passed on the morning of 2026-09-04.

## Reviewed evidence, per locale

Source-bound records under `docs/i18n/reviewed/runtime/<locale>/`, against the
5,165-entry runtime denominator. As before, repeated source strings mean this
is not a unique-string percentage, and the proportion is small by design: the
evidence is defect-driven.

| Language | Reviewed runtime records | Of 5,165 |
|---|---:|---:|
| Swedish | 117 | 2.27% |
| German | 86 | 1.67% |
| Spanish | 93 | 1.80% |
| Simplified Chinese | 240 | 4.65% |
| Portuguese | 96 | 1.86% |
| Hindi | 107 | 2.07% |
| Korean | 233 | 4.51% |
| Icelandic | 121 | 2.34% |
| French | 96 | 1.86% |

## Who reviewed what, and what that claim means

**The maintainer (Einar Olafsson) reads Icelandic, Swedish and German**, and
said so explicitly when asked on 2026-09-02. Only those three locales carry any
human-native review, and only for the strings he was shown.

**On 2026-09-04 he directed that the remaining pass be done and recorded as
reviewed.** That decision is honoured, and its provenance is recorded rather
than blurred: rows added that day were drafted by Claude and accepted by the
maintainer. `docs/i18n/reviewed/runtime/*/2026-09-04-load-family.json` carries
`drafted_by`, `accepted_by` and an `acceptance` note saying in terms that this
is a maintainer's acceptance and **not** a native-speaker review, and that he
did not read fr, hi, ko or zh_CN.

**The honest public claim remains the one 316 proposed:** "nine languages,
machine-drafted and technically reviewed" is defensible. "Translated into nine
languages" is not, and should not be written in the README, the paper or the
Zenodo record until a reader per locale has passed over them.

## Explicit fallbacks and allowlisted untranslated terms

* `build_i18n_catalogs._IDENTITY_TEXT` — 84 terms that stay English by
  decision (GPU, CSV, gRNA, Amsgrad, Eps and similar). These are excluded from
  the exact-English gate through `_looks_translatable`, which is why a naive
  equality count of "untranslated" rows overstates the work by roughly fifty
  times; measured with the gate's own filter the true remainder was 16, and it
  is now 0.
* `tools/i18n_reviewed_ui.py` — 87 context-pinned UI sources, of which 21 are
  **reviewed identities**: the correct translation is byte-identical to the
  English. `fr: "Source"`, `fr: "Figure"`, `de: "Well"`, `de: "Wells"` and
  `sv/de/pt: "Gate"` are of this kind, and each is an explicit decision rather
  than an untranslated leftover.
* `API_EXACT_TEXT_ALLOWLIST` in `tests/qt/test_external_i18n_catalogs.py` — 17
  API symbols whose rendered text is legitimately identical in some locale.
  Audited entry by entry on 2026-09-04 and reduced from 19: one entry was
  translated in every locale and one named a symbol that no longer exists.

## Defects found and their disposition

Every defect found on 2026-09-04 is listed with what was done about it.

| Defect | Locales | Disposition |
|---|---|---|
| "load" rendered as 충전, to recharge a battery | ko (38 strings) | repaired in the visible label family |
| "load" rendered as "Láttu upp" (let/make) | is | repaired |
| `'Load a column mapping'` rendered "폴더 폴더 폴더 폴더" | ko | repaired |
| "Object" rendered "Syfte" (purpose) | sv | repaired |
| "Model zoo" rendered 动物园模型 (a model of a zoo) | zh_CN | repaired to 模型库 |
| plaque family rendered 板 / 板块 (board, tectonic plate) | zh_CN | repaired to 蚀斑 |
| a microplate well rendered as a water well, the sea, or the adverb | de 5, is 10, ko 2, hi 4 | 21 rows retranslated whole |
| composed setting names left their suffix untranslated in every locale | all nine | fixed in `_composite_translation` |
| `intermedeate_save` still asserted the setting "has no effect" after the English was rewritten | all nine | retranslated |
| `SETTING_LABELS['src']` rendered as a road (ko 도로), a lane (fr Voie), a route (is leiðin) | all nine | retranslated to the Source term |
| "call fields" rendered as TELEPHONE fields | ko, zh_CN (API catalog) | recorded, not yet repaired — see below |

## What is knowingly not done

* **The API catalog's own false friends.** `spacr.regression_panels.apply_primary_call`
  renders "call fields" as 전화 필드 (ko) and 电话字段 (zh_CN) — a telephone call,
  where the sense is a statistical call. It is fluent, passes every gate, and is
  in the API catalog rather than the runtime one, which is a surface no
  systematic reading has covered. Filed against instruction 316.
* **A per-locale fluent reader.** Six of nine locales have never had one, and
  the 2026-09-04 acceptance does not change that.
