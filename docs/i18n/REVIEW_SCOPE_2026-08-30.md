# Localization review scope at the 1.5.0.5 release gate

This is an evidence report, not a certificate that every sentence was read by
a fluent speaker.  The catalogs have complete mechanical source coverage, but
the source-bound semantic-review evidence is defect-driven and much smaller
than the shipped corpus.  Instruction 306 therefore must not be closed on the
basis of these numbers alone.

## Inventory and evidence scope

The live runtime inventory contains 4,960 entries: 1,002 setting labels, 997
setting tooltips, 192 category explanations, 2,705 UI strings, and 64 module
summaries.  The live API inventory contains 8,899 symbol documents, including
119 rendered aliases.  The nine supported non-English locales are Swedish
(`sv`), German (`de`), Spanish (`es`), Simplified Chinese (`zh_CN`), Portuguese
(`pt`), Hindi (`hi`), Korean (`ko`), Icelandic (`is`), and French (`fr`).

The table counts unique, current source strings admitted through
`docs/i18n/reviewed/runtime/<locale>/` and unique, current translatable blocks
admitted through `docs/i18n/reviewed/api/<locale>/`.  Percentages and remainders
use the requested release denominators (4,960 runtime entries and 8,899 API
symbol documents).  API evidence is block-keyed, not document-keyed, so its
percentage is an intentionally conservative normalization, not the proportion
of whole API documents read.  Repeated source strings likewise mean that the
runtime percentage is not a unique-string percentage.

| Language | Reviewed runtime sources | Of 4,960 | Arithmetic remainder | Reviewed API blocks | Of 8,899 | Arithmetic remainder |
|---|---:|---:|---:|---:|---:|---:|
| Swedish | 78 | 1.57% | 4,882 | 180 | 2.02% | 8,719 |
| German | 48 | 0.97% | 4,912 | 122 | 1.37% | 8,777 |
| Spanish | 55 | 1.11% | 4,905 | 16 | 0.18% | 8,883 |
| Simplified Chinese | 202 | 4.07% | 4,758 | 140 | 1.57% | 8,759 |
| Portuguese | 59 | 1.19% | 4,901 | 128 | 1.44% | 8,771 |
| Hindi | 67 | 1.35% | 4,893 | 74 | 0.83% | 8,825 |
| Korean | 183 | 3.69% | 4,777 | 106 | 1.19% | 8,793 |
| Icelandic | 73 | 1.47% | 4,887 | 588 | 6.61% | 8,311 |
| French | 56 | 1.13% | 4,904 | 106 | 1.19% | 8,793 |

In addition, `tools/i18n_reviewed_ui.py` pins 84 context-sensitive UI sources
in every locale: 84 x 9 = 756 reviewed source/target pairs.  Those rows overlap
the runtime inventory and are therefore not added to the table.  The review
evidence schemas do not consistently name a fluent-speaker reviewer; some
records and the Portuguese API evidence explicitly describe Codex-assisted
review.  No fluent-speaker census of all 44,640 runtime target entries or all
80,091 API target documents is recorded.

## Sampling method and exact unsampled remainder

Review was purposive, not random: model-rejected hard tails, scientific false
friends, source changes, known UI ambiguity, former tooltip waivers, and defects
found while rendering were selected.  The arithmetic remainders in the table
are exact against the requested denominators, but they are not proof that each
remaining entry is wrong or wholly unreviewed: one reviewed API block can occur
in several documents, and static reviewed vocabularies overlap external review
files.  There was no unnamed statistical sample and no claim of exhaustive
semantic review.

The final source-bound repair reviewed all four blocks of
`spacr.__main__.main` in each language: two unchanged source blocks plus the
new successful process-exit-code and `SystemExit(2)` parser-error contract
blocks.  It performed no model decoding.

## Explicit English identities and fallbacks

The runtime catalogs preserve this complete 84-item reviewed technical-identity
set exactly in English where it occurs:

`%d px`; `--partition=gpu --gres=gpu:1 --time=12:00:00`; `2D / 3D UMAP`;
`3D`; `<a href="api">API</a>`; `API`; `Amsgrad`; `CPU`; `CSV`; `CUDA`; `CV`;
`Cellpose-SAM`; `DNA`; `EAF1_g1, EAF1_g2`; `EC50`; `Eps`; `FOV`; `FlowView`;
`GPU`; `Huber t`; `JSON`; `MIP`; `ML`; `NaN`; `PDF`; `PNG`; `QC`; `RGB`;
`RNA`; `ROI`; `RdBu_r`; `SAM`; `SHAP`; `SQL`; `TIFF`; `Tensorboard`; `UMAP`;
`ViT`; `X`; `XGBoost`; `Y`; `Z`; `btrack`; `cellpose`; `cividis`; `coolwarm`;
`dst`; `fdr_bh`; `gRNA`; `gRNA CSV`; `image_path`; `inferno`; `log10`; `magma`;
`measurements.db`; `metadata_column_map.json`; `numpyro`; `otsu`; `plasma`;
`png`; `png_list`; `png_path`; `pymc`; `seg_qc`; `slurm`; `spaCR`; `ssh`; `t`;
`torch`; `trackastra`; `trackpy`; `tsne`; `turbo`; `ultrack`; `umap`; `viridis`;
`x`; `xD`; `y`; `{report}`; `|Tutorials|`; `µM`; `µm/pixel`; `■ {note}`.

Reviewed locale-specific spellings that are also byte-identical to English are:

- Swedish: `Toxoplasma`, `Regex`, `Budget`, `Gate`.
- German: `Toxoplasma`, `Regex`, `Radius`, `Folds`, `Budget`, `Gate`, `Well`,
  `Wells`.
- Spanish: `Toxoplasma`, `Regex`, `Coef.`.
- Simplified Chinese: `Regex`.
- Portuguese: `Toxoplasma`, `Regex`, `Gate`, `Box gate`, `Coef.`.
- Hindi: `Regex`.
- Korean: `Regex`.
- Icelandic: `Toxoplasma`, `Regex`.
- French: `Toxoplasma`, `Regex`, `Axes…`, `Budget`, `Gate`, `Figure`, `Type`.

The API catalogs preserve exact English only for these 19 reviewed symbol
documents, whose visible contents are code, identifiers, literals, or data
shapes rather than translatable prose:

- `spacr.align.CanvasSpec.shape`
- `spacr.errors.RunLedger.status`
- `spacr.gene_facts.Segment.text`
- `spacr.hits.HitList.flag_counts`
- `spacr.macro.MacroStep.entry`
- `spacr.qt.iconset.themed_pixmap`
- `spacr.qt.screens.report.ReportScreen.output_format`
- `spacr.qt.settings_search.SettingsSearchBar.level`
- `spacr.qt.widgets.dose_response.DoseResponseResult.status`
- `spacr.qt.widgets.formula.Unary`
- `spacr.qt.widgets.plate_layout.PlateDesign.shape`
- `spacr.qt.widgets.setup_card.SetupCard.mode`
- `spacr.resources.home.versions._generators.common.app_map`
- `spacr.run_compare.HitList.by_key`
- `spacr.runctx.RunContext.__str__`
- `spacr.runctx.SkipRecord.__str__`
- `spacr.schema.field_index`
- `spacr.seg_qc.Scorecard.verdict`
- `spacr.updater.PackageChange.describe`

There are no source-current exact-English setting-tooltip prose rows.  Unknown
runtime strings, stale runtime records, malformed/stale API payloads, and API
fetch failures fall back visibly to canonical English and are not counted as
translations.  At this snapshot there are no unresolved catalog rows using
that fallback; the fallback remains a tested safety path.

## Mechanical and rendered evidence

Both catalog audits enforce live-source keys and hashes, placeholder/markup and
protected-literal parity, expected target scripts, copied-English residue, and
known semantic false-friend families.  Required docs CI runs a clean
`sphinx-build -W -E --keep-going`, preserves Sphinx's exit status, and resolves
every emitted tooltip API link against the fresh output.

Runtime tests drive real catalogs through Qt language switching.  Browser tests
exercise rendering, selector persistence, malformed/stale payload fallback, and
request races with two-symbol synthetic catalogs.  They do not drive all nine
real 8,899-document catalogs through a browser.  Synthetic source-staleness and
partial-Sphinx failures are covered; there is no dedicated mutation test that
injects each of a missing tooltip translation and a broken tooltip API link.
Those are closure gaps, not successes hidden by the green fixed-tree tests.

The former twelve `KNOWN_THIN` tooltip waivers are gone.  Each now passes the
ordinary type, length, non-tautology, default-presence, and link checks and is
separately pinned to implementation facts.  The suite does not mechanically
prove unit presence, conditional-applicability accuracy, or equality between
every tooltip's stated default and every module-specific live default.  API
parameter checks reject ghost `:param:` names and contradictory literal
defaults, but do not require every real required public parameter to have a
field.

## Disposition

The source/catalog fixed point and warning-as-error documentation gate can be
green while the semantic acceptance criteria above remain incomplete.  Item
306 can move to done only after the remaining rendered real-catalog exercise,
explicit synthetic mutation ratchets, tooltip unit/default/applicability
invariants, bidirectional public-parameter audit, and a named exhaustive or
exactly sampled semantic review are completed.
