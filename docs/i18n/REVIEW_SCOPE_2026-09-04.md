# Localization review scope at the 1.5.0.5 release gate — 2026-09-04

This supersedes `REVIEW_SCOPE_2026-08-30.md`, which is kept as history. Like
that report, **this is an evidence report and not a certificate that every
sentence was read by a fluent speaker.** The semantic-review evidence is
defect-driven and much smaller than the shipped corpus.

**Mechanical source coverage IS complete, as of 2026-09-10.** It was not on
2026-09-06, and this paragraph carried that gap until now: the runtime
catalogs held 5,105 of 5,226 entries in every locale -- 121 sources missing
each, 1,089 in total -- and the API catalogs 8,966 of 10,241. The machine
that measured it had no OPUS checkpoint, so the gap was recorded rather than
closed.

It is closed. Both audits pass on the same tree:

    verified API catalogs: languages=9 symbols=10306
    verified external runtime catalogs: languages=9 settings=1073
      categories=201 ui=2841 modules=67

and `docs/i18n/COVERAGE.md`, regenerated beside this file, reports
`5260/5260` runtime and `10296/10296` API for every one of the nine, with
zero orphans. Two hand-written records were needed where the model could not
satisfy the gates: `spacr.ops_accel#4` in Portuguese and the `ops_gpu`
tooltip in Simplified Chinese.

WHAT REMAINS IS THE REVIEW, WHICH IS THE POINT OF THIS FILE. Mechanical
coverage means every source has A translation. It does not mean a speaker of
the language has read it, and the table below is what says how much of that
has actually happened -- between 1.6 % and 9.9 % per locale.

## What changed since 2026-08-30

The runtime inventory grew from 4,982 entries to **5,165**: 1,019 setting
labels, 1,014 setting tooltips, 192 category explanations, 2,817 UI strings, 57
installer strings and 66 module summaries. The API inventory grew from 8,861
symbol documents to **9,434**, all ten locales in agreement.

Every mechanical gate now passes. `tests/qt/test_external_i18n_catalogs.py` is
31 of 31, having been 7 failed / 24 passed on the morning of 2026-09-04.

## Reviewed evidence, per locale

Source-bound records under `docs/i18n/reviewed/runtime/<locale>/` and
`docs/i18n/reviewed/api/<locale>/`, against the LIVE denominators rather than a
remembered one: 5,249 runtime entries and 10,339 public API docstrings. As before,
repeated source strings mean this is not a unique-string percentage, and the
proportion is small by design: the evidence is defect-driven.

Both denominators move whenever a string or a docstring is added, so these
numbers are regenerated rather than transcribed; the test that guards this
table derives them from the same source the builders read.

| Language | Reviewed runtime records | Of 5,662 | Remainder | Reviewed API blocks | Of 10,533 | Remainder |
|---|---:|---:|---:|---:|---:|---:|
| Swedish | 118 | 2.08% | 5,544 | 468 | 4.44% | 10,065 |
| German | 85 | 1.50% | 5,577 | 420 | 3.99% | 10,113 |
| Spanish | 99 | 1.75% | 5,563 | 287 | 2.72% | 10,246 |
| Simplified Chinese | 267 | 4.72% | 5,395 | 507 | 4.81% | 10,026 |
| Portuguese | 99 | 1.75% | 5,563 | 446 | 4.23% | 10,087 |
| Hindi | 128 | 2.26% | 5,534 | 460 | 4.37% | 10,073 |
| Korean | 246 | 4.34% | 5,416 | 453 | 4.30% | 10,080 |
| Icelandic | 151 | 2.67% | 5,511 | 1,021 | 9.69% | 9,512 |
| French | 98 | 1.73% | 5,564 | 465 | 4.41% | 10,068 |

*Regenerated 2026-09-14, second pass, for instruction 364's `grna`
retirement. The runtime denominator moves 5,664 -> 5,662 and NOT by a flat
two deletions: `grna`'s setting label and its tooltip leave (-2), the
`PATH_LIST_TITLES` chooser caption 'Choose the gRNA CSV' leaves with the
control it named (-1), and the identity 'gRNA' ARRIVES as a UI record (+1).
It arrives because of the removal: while `grna` was a setting whose English
label was 'gRNA', that identity was materialised by the label table and
excluded from the UI sources; retiring the setting hands it to UI instead.
NO REVIEWED RECORD CHANGED in any locale -- the nine reviewed counts are
exactly what they were, because `grna` had no reviewed record in any
language. Its sibling `barcodes` DOES (zh_CN, `setting_tooltips`), which is
why `barcodes` is approved for the same retirement and is not in this pass.
Only zh_CN's percentage moves, 4.71% -> 4.72%, and that is 267 divided by a
denominator two smaller rather than any change in what has been read.*

*Regenerated 2026-09-14. The API denominator moved 10,478 -> 10,531 as well:
53 public symbols arrived and none left, set-differenced against the
baseline rather than subtracted -- 44 from the new curation-queue modules,
7 from the default-graph-type work and 2 from the figure-font work.
The runtime denominator moved 5,361 -> 5,664: 283
captions reached the extractor through fourteen runtime registries that an AST
walk could only see as a variable, and 21 more were the regression-model menu,
which `settings_model` DECLARED for exactly this purpose and which nothing
consumed -- so all 21 read English in all nine locales. Three API counts fall
by one or two (de 424 -> 422, pt 448 -> 446, is 1,025 -> 1,022): protecting the
twenty UI screen names invalidated reviewed records that had TRANSLATED a
screen name, which sent a reader to a screen their interface does not have.
Those records were wrong the day they were written; nothing could see it until
the term was protected. Two locales gain a runtime record (zh_CN 266 -> 267,
hi 126 -> 128) for the three em-dash definition rows no model would take.*

*Regenerated 2026-09-13. The runtime denominator moved 5,244 -> 5,361 because
a one-line `_set_status` wrapper had hidden 117 user-facing status captions
from the extractor across twenty-three screens; they are now declared and
translated in all nine locales. NO REVIEWED RECORD CHANGED -- the reviewed
counts are what they were, and the percentages fall only because the surface
they are measured against grew. That is the honest direction for this number
to move: the captions were always there and always unreviewed, and the report
now counts them.*

*Regenerated 2026-09-11 against the tree after the merge from `main`. Both
denominators moved -- the runtime one DOWN, 5,260 to 5,249, because 364
deleted settings and their captions went with them, and the API one UP,
10,306 to 10,339, with the modules 370, 372, 377, 386, 387 and 388 added.
Several locales lose a record or two for the same reason a setting's caption
left: a reviewed record whose source is no longer in the catalog is not
evidence about anything. The one addition is `press Escape to close`, claimed
in all nine after a rebuild had replaced the key name with a verb in eight of
them.*

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

---

## Table regenerated 2026-09-13

The denominators moved with the Map Barcodes merge and the week's settings
work: runtime 5,265 -> 5,244 sources, API 10,400 -> 10,478 symbols. The test
that guards this table derives both from the builders, so it went red until
this was regenerated -- which is the contract, not a defect.

**SEVERAL REVIEWED COUNTS FELL, and that was checked rather than assumed.**
Spanish 114 -> 99, Portuguese 107 -> 99, Korean 250 -> 245, Chinese 266 -> 262.
A falling count of reviewed records is the signature of STRANDED EVIDENCE -- a
record keyed to a source that has been renamed or removed -- which has happened
in this repository before and cost 32 records.

It is not what happened here:

  * `tools/check_reviewed_runtime_evidence.py` reports every reviewed runtime
    record still matching its source, 1,287 checked, exit 0.
  * The catalogs' reviewed sections were not touched by the 2026-09-13
    rebuild or repair at all: `git diff` on `es`, `pt`, `ko` and `zh_CN`
    contains ZERO lines mentioning `reviewed`.

So the fall happened between 2026-09-04 and now, as 391 retired settings and
renamed others, and the records for settings that no longer exist went with
them. The evidence was not lost; the things it was evidence ABOUT were
withdrawn.

Reviewed totals today: sv 118, de 85, es 99, zh_CN 267, pt 99, hi 128, ko 246, is 151, fr 98 -- 1,291 runtime records across nine locales.

