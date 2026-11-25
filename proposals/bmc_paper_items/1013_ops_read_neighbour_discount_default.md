# Item 1013: neighbour-bleed read discount 0.25 as the default (`ops_read_neighbour_discount` 1.0 → 0.25)

- **Number:** 1013 (master, 2026-11-21). **Settings schema:** 956 (`DEFAULT_CHANGES[956]`, `SETTINGS_SCHEMA_VERSION = 956`). Item 1014 follows at 957.
- **Decided by:** the maintainer, 2026-11-21 ~15:00 (decision 2), with a figure panel that motivates the value.
- **Code basis:** origin/nightly 19a95cd799. Spec only; nothing here is implemented.
- **Repo rule 43 (one operation, one code path):** only the default flips. The discount is the item-988 rule: inside a cell, a guide with strictly fewer reads than the top guide is multiplied by b when it is the top guide (by reads, ties kept) of a neighbouring cell. §2 records that this rule exists TWICE today, so this item also specifies the one shared core (§3). The core is a separate lane; the default flip does not wait for it (§6).

## 0. Evidence (item 988 acceptance; script harness around spaCR's own decode)
**Funk et al. plate A, A1** (333 fields, 1,279,317 nuclei; held-out report half: 167 fields, 633,292 nuclei; paired field bootstrap 1,000; `reanalysis/item988_acceptance/tables/funk/funk_arms.csv`, `funk_verdict.json`, `funk_shuffle_500.csv`):

| b | Δ genotyped at matched FDR (points) [95 % CI] | Δ decoy FDR at matched (%) [95 % CI] | rescued / lost cells | agreement with b = 1 |
|---|---|---|---|---|
| 1.0 (today, reference) | 0 | 0 | 0 / 0 | 100 % |
| 0.5 | +0.31 [0.29, 0.33] | [−0.0004, +0.0002] | 2,029 / 71 | 99.84 % |
| **0.25** | **+0.38 [0.37, 0.40]** | **[−0.0004, +0.0004]** | 2,462 / 24 | 99.80 % |
| 0.0 | +0.43 [0.41, 0.45] at production, but the FDR rises: [+0.00004, +0.0011] (production), [+0.0003, +0.0014] (knee) | | 2,734 / 15 | 99.78 % |

- **Why 0.25:** it is the largest gain whose paired FDR change includes 0 at both the production point and the knee. b = 0 adds yield but raises the decoy FDR, because ignoring a minority guide entirely lets a single stray read decide.
- **Pre-stated 988 verdict for N1 b = 0.25: PASS.** The criteria are yield at the knee (judged by the paired CI) and at matched FDR, FDR relative to baseline, all four edge-distance quartiles positive, and agreement ≥ 99.5 %.
- **Shuffle call rate** (500 seeds, production point): 0.8634 % [0.753, 0.989] against 0.8634 % for b = 1. The difference is +0.000008 points.
- **By edge-distance quartile** (Δ genotyped, points [CI], 158,323 cells each): Q1 +0.35 [0.32, 0.38], Q2 +0.36 [0.34, 0.39], Q3 +0.40 [0.37, 0.44], Q4 +0.42 [0.39, 0.46].

**PERISCOPE CP186A W1, PERISCOPE masks** (43 held-out sites; `tables/periscope/periscope_arms.csv`):
- b = 0.25 under the weighted majority (+ quality) gains +1.60 points [1.26, 1.98] at matched FDR over the recipe_simple + 967 k3 reference, with Δ FDR [−0.0041, −0.0003].
- Quality-only weighted majority (b = 1) gains +0.40 [0.22, 0.57].
- So b = 0.25 adds about 1.2 points on top of the method change. **But see §2: the default PERISCOPE method is recipe_simple, where b does not act.**

**Figures (paper repo, built 2026-11-21):**
- Supplementary Figure S2 **M**, "why b = 0.25 is the default": the b curve on Funk et al. A1 and Ramezani et al. W1. It shows Δ genotyped at matched FDR and Δ decoy FDR at the knee against b, with paired CIs. The chosen b is circled.
- Supplementary Figure S2 **G**: all 988 arms. Its shuffle-rate axis is now filled from the 500-seed run.
- Development Figure D9 **F**: Δ genotyped per edge-distance quartile.
- Choice-to-panel register row (`reanalysis/figures_choice_register.md`): "Neighbour-bleed discount default b = 0.25 → S2 M + S2 G + D9 F".

## 1. Setting and DEFAULT_CHANGES
- `ops_read_neighbour_discount`: default **0.25** (was 1.0), in both `ops_settings.py` (~232) and `ops_decode.py` (`_DEFAULTS` ~100).
- The tooltip (~3202) changes "Default 1. Alpha." to "Default 0.25". It stays a visible control: it is a default now, not alpha. `ALPHA_FEATURES[988]` loses this one key, and its other keys (`ops_read_distance_*`) stay alpha.
- Row:
  ```python
  956: {'ops_read_neighbour_discount': 1.0},
  ```
  `SETTINGS_SCHEMA_VERSION = 956`. Docstring paragraph: "956 is item 1013's neighbour-bleed discount 0.25 (item 988); files stamped before keep 1.0 (off)." Add a line to `docs/source/reproducibility_manifests.rst`.

## 2. Where reads are assigned, and whether the discount reaches them (checked at 19a95cd799)
| Path | Reads assigned by | Discount code | Neighbours | Acts under the default settings? |
|---|---|---|---|---|
| Funk full-barcode, rule (a)/S2 (`ops_decode.decode_rule` → `_field_votes` → `ops_sbs.vote_mapped_reads`) | `vote_mapped_reads` | `ops_sbs._discount_neighbour_bleed` (dict-based, ~1103) | `ops_sbs._disc_neighbours`: equal-area nucleus discs within `ops_footprint` (~1135), built by `ops_engine` (~3154) | Yes, when the decode rules are active (`rule["active"]`: decoys or a rule switch on). The Funk production preset is active |
| Funk cache replay (`_joint_field_candidates`, decode QC, auto tuner) | the same | the same | `_rule_with_neighbours` / `cached_neighbours`; a cache without nucleus positions raises (`ops_decode.py` 626, 2037) | Raises with b ≠ 1 when the cache lacks positions |
| Spacer-prefix (PERISCOPE) weighted majority (`ops_prefix._weighted_counts`, `_weighted_cells`) | `_weighted_decision` | `ops_prefix._barcode_groups` bleed flag + `_discounted` (vectorised, ~897/960) | `ops_prefix._touching_cells`: 4-connected touching cell labels (~779) | Only with `ops_prefix_quality_method = 'weighted_majority'` |
| Spacer-prefix recipe_simple / 'spacr' (the default method) | `prefix_cell_calls` → `recipe_simple_quality` / `classify_cell` | none | n/a | **No**, and validation (~2162) **raises** for b ≠ 1 without weighted majority |
| Legacy (rules off) full-barcode vote | `vote_mapped_reads` without rule | none | n/a | **No**, and `decode_rule` (~238) **raises** for b ≠ 1 with rules off |

So:
1. **The PERISCOPE production path does not use the discount.** Its default `recipe_simple` has no vote to weight.
2. **The bleed rule exists twice**, the dict version in `ops_sbs` and the vectorised version in `ops_prefix`, with the same definition and different code.
3. **A bare default flip would break runs.** Every prefix run with the default method, every rules-off full-barcode run, and every cache replay without positions would raise on the new default.

## 3. Behaviour after 1013
1. **Validation for a default that does not apply.** The two "raises for b ≠ 1" checks (`ops_decode.py` ~238, `ops_prefix.py` ~2162) become, only when b equals the *default* 0.25: the discount is not applied on that path, and one notice per run goes to the log (`tr`): "ops_read_neighbour_discount (0.25) does not act under <method/rules off>; reads are not discounted". An explicitly entered b different from both 1 and the default still raises as today. The resolved run journal records `ops_read_neighbour_discount_applied: true/false` per path, so a rerun is exact.
2. **Cache replay without nucleus positions** (decode QC, ops_live, compare_calls on old caches): same rule. With the default, replay runs undiscounted, with a notice and `applied: false`. With an explicit b it raises as now. New caches already store positions (`cached_neighbours`).
3. **PERISCOPE shared use (specified here, gated).**
   - Under `recipe_simple`, the shared bleed flag (§4) marks a neighbour's top barcode in a cell's minority. With b < 1, a flagged minority barcode is left out exactly as `_minority_ignored` leaves out a minority below `ops_prefix_minority_min_reads` (item 967), so a cell whose only second barcode is neighbour bleed becomes single-barcode. This is the recipe_simple reading of "weigh the bleed down"; under a counting method, any b < 1 acts as 0. It is a NEW behaviour on the PERISCOPE production method.
   - It ships behind its own alpha switch, `ops_prefix_bleed_in_recipe` (default False, `ALPHA_FEATURES[1013]`), until a PERISCOPE acceptance passes. That acceptance uses recipe_simple + k3 against the same + bleed flag on CP186A W1 PERISCOPE masks, with the same 988 criteria and both error rates, through the paper lane.
   - The flag becomes default only by a later DEFAULT_CHANGES row. Until then PERISCOPE runs with `applied: false` and the notice, which is honest.
   - The alternative, making `weighted_majority` the prefix default (+1.60 points with b 0.25 in §0), is a method change for the maintainer and is NOT part of this item.

## 4. One shared bleed core (rule 43; lane L2, does not block L1)
- New module-level function `ops_sbs.neighbour_bleed_flags(cell, barcode, reads, neighbours) -> bool per (cell, barcode) group`. It is the vectorised rule of `ops_prefix._barcode_groups`, moved. Inputs are integer cell codes, barcodes, per-group read counts and the `(P, 2)` neighbour pairs.
- `ops_sbs._discount_neighbour_bleed` becomes a thin adapter: dict tallies → arrays → the flags → multiply. `ops_prefix._barcode_groups` calls the same function for its `bleed` column. `_discounted` is unchanged.
- Neighbour sources stay input-specific by design: nucleus discs within `ops_footprint` for the full-barcode decode, touching cell labels for the prefix decode. Both produce the same `(P, 2)` pairs format (`_disc_neighbours` adds a dict → pairs converter). The tooltip already states the two sources.
- **Equivalence test:** on 20 Funk A1 cached fields and 10 PERISCOPE W1 sites, the votes and calls with the shared core are identical to today's for b ∈ {1, 0.5, 0.25, 0}.

## 5. Tests (`tests/test_ops_neighbour_discount_default_1013.py`; CUDA_VISIBLE_DEVICES='' -n 4)
1. The default is 0.25 in both default maps. `DEFAULT_CHANGES[956]` makes an unstamped or 955 file load 1.0, and its decode tables are byte-identical to today's (`cli_repro`).
2. Prefix with the default `recipe_simple`, a rules-off full-barcode run and a position-less cache replay: no error, one notice, `applied: false`, calls identical to b = 1. An explicit b = 0.5 there still raises.
3. Funk rule (a) on the OPS test fixture (item 863 A1 subset): b = 0.25 changes calls exactly as `_discount_neighbour_bleed` does (regression against the 988 tests, which pass unchanged).
4. §4 equivalence test (lane L2).
5. `ops_prefix_bleed_in_recipe`: off changes nothing. On, a synthetic cell with a neighbour's top barcode as its only minority becomes single-barcode. The switch is registered in `ALPHA_FEATURES[1013]`.
6. Ratchets before hand-over: raw-reader, nested-function, documented-callable (`neighbour_bleed_flags` ships with arrival contract + 9-locale records in the same commit) and hash-comment.

## 6. Lane split (file boundaries)
About 120–180 changed lines plus about 200 test lines. Half a day for L1; half a day for L2.

| Lane | Owner | Files (exclusive) | Blocks the pilot? |
|---|---|---|---|
| **L1 default flip** | WS/Codex P1 | `spacr/ops_settings.py` (default, tooltip), `spacr/ops_decode.py` (`_DEFAULTS`, validation ~238, cache-replay rule ~626/2037, `applied` record), `spacr/ops_prefix.py` (validation ~2162 only), `spacr/settings.py` (`DEFAULT_CHANGES[956]`, schema, `ALPHA_FEATURES` edits), `docs/source/reproducibility_manifests.rst`, tests §5.1–5.3 | yes (the Funk pilot uses b = 0.25) |
| **L2 shared core + PERISCOPE switch** | WS/Codex | `spacr/ops_sbs.py` (`neighbour_bleed_flags`, adapter), `spacr/ops_prefix.py` (`_barcode_groups` call, `ops_prefix_bleed_in_recipe` in `_minority_ignored` path), tests §5.4–5.5 | no |
| **L3 translations** | Codex tr | `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_1013_task.md` (tooltip, notice, new switch; sources with SHA-256, 9 locales) → `spacr/qt/i18n_catalogs/*.py` after L1/L2 freeze the strings | no |
| **L4 acceptance** | Home paper lane | Funk A1 rerun with the spaCR default equals the 988 N1_b0.25 replay (974,437 → the b = 0.25 call set, to the cell); PERISCOPE recipe_simple + bleed-flag acceptance (§3.3); no `spacr/` files | Funk part yes |

Conflicts:
- **With 1014:** `DEFAULT_CHANGES` rows 956/957 and `SETTINGS_SCHEMA_VERSION`. These are one-line rebases; the later lander takes the higher number.
- **With 1011 (955):** the same.
